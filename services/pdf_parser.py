import base64
import importlib
import io
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_LAYOUT_PARSE_SYSTEM_PROMPT = (
    "You are a precise document parser. Convert the provided PDF page image into faithful markdown. "
    "Preserve heading hierarchy, lists, equations (as plain text or markdown), and especially tables "
    "using markdown table syntax. Do not summarize, do not omit details, and do not add extra commentary. "
    "If figures/charts are visible, insert placeholder tokens in reading order like [[IMAGE_P{page}_I1]], "
    "[[IMAGE_P{page}_I2]], etc. Return markdown only."
)

_IMAGE_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a technical analyst for engineering and scientific documents. "
    "Describe the provided figure/chart/image in a rigorous and compact way. "
    "Mention axes/units/trends for charts, table-like data patterns if visible, "
    "and any conclusions that can be directly observed. Do not hallucinate unreadable values."
)

_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?[-]{3,}:?\s*(\|\s*:?[-]{3,}:?\s*)+\|?\s*$")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PAGE_HEADER_RE = re.compile(r"\bpage\s+(\d{1,5})\b", re.IGNORECASE)
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[\[IMAGE_P(\d+)_I(\d+)\]\]")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^\)]+\)")


def _log_error(log_error: Optional[Callable[[str, Exception], None]], context: str, exc: Exception) -> None:
    if callable(log_error):
        log_error(context, exc)


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(parsed, maximum))


def _coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(minimum, min(parsed, maximum))


def _strip_markdown_fences(text: str) -> str:
    rendered = str(text or "").strip()
    if not rendered:
        return ""

    if rendered.startswith("```"):
        lines = rendered.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        rendered = "\n".join(lines).strip()

    return rendered


def _normalize_header_path(parts: Iterable[str]) -> str:
    cleaned = [str(item).strip() for item in parts if str(item).strip()]
    return " > ".join(cleaned)


def _extract_page_number_from_header_path(header_path: str) -> Optional[int]:
    match = _PAGE_HEADER_RE.search(str(header_path or ""))
    if not match:
        return None
    try:
        page_number = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return page_number if page_number > 0 else None


def _contains_table(text: str) -> bool:
    lines = str(text or "").splitlines()
    if len(lines) < 2:
        return False

    for index in range(len(lines) - 1):
        if _TABLE_LINE_RE.match(lines[index]) and _TABLE_SEPARATOR_RE.match(lines[index + 1]):
            return True
    return False


def _contains_image_description(text: str) -> bool:
    rendered = str(text or "")
    return "Image Description (Page" in rendered


def _content_type_for_text(text: str) -> str:
    if _contains_image_description(text):
        return "image"
    if _contains_table(text):
        return "table"
    return "text"


def _split_atomic_blocks(text: str) -> List[str]:
    rendered = str(text or "").strip()
    if not rendered:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", rendered) if part.strip()]
    blocks: List[str] = []
    buffer: List[str] = []
    in_table = False

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            blocks.append("\n\n".join(buffer).strip())
            buffer = []

    for part in paragraphs:
        is_table_part = _contains_table(part)
        is_image_desc = _contains_image_description(part)

        if is_image_desc:
            flush_buffer()
            blocks.append(part)
            in_table = False
            continue

        if is_table_part:
            if not in_table:
                flush_buffer()
                buffer = [part]
                in_table = True
            else:
                buffer.append(part)
            continue

        if in_table:
            flush_buffer()
            in_table = False

        blocks.append(part)

    flush_buffer()
    return [item for item in blocks if item.strip()]


def _chunk_text_without_breaking_atomic_blocks(text: str, max_chars: int) -> List[str]:
    blocks = _split_atomic_blocks(text)
    if not blocks:
        return []

    cap = max(500, int(max_chars))
    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_len = 0

    for block in blocks:
        block_len = len(block)

        if block_len > cap:
            flush()
            chunks.append(block)
            continue

        projected = block_len if not current_parts else current_len + 2 + block_len
        if current_parts and projected > cap:
            flush()

        if not current_parts:
            current_parts = [block]
            current_len = block_len
        else:
            current_parts.append(block)
            current_len += 2 + block_len

    flush()
    return [item for item in chunks if item.strip()]


def _fallback_markdown_sections(markdown_text: str) -> List[Dict[str, Any]]:
    rendered = str(markdown_text or "").strip()
    if not rendered:
        return []

    sections: List[Dict[str, Any]] = []
    current_headers: List[str] = []
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_lines
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(
                {
                    "page_content": section_text,
                    "metadata": {
                        "header_path": _normalize_header_path(current_headers),
                    },
                }
            )
        current_lines = []

    for line in rendered.splitlines():
        header_match = _HEADER_RE.match(line.strip())
        if header_match:
            flush()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_headers = current_headers[: level - 1]
            current_headers.append(title)
            current_lines = [line]
            continue
        current_lines.append(line)

    flush()
    return sections


def _split_markdown_by_headers(markdown_text: str) -> List[Dict[str, Any]]:
    rendered = str(markdown_text or "").strip()
    if not rendered:
        return []

    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
            strip_headers=False,
        )
        documents = splitter.split_text(rendered)
        sections: List[Dict[str, Any]] = []

        for doc in documents:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            page_content = str(getattr(doc, "page_content", "") or "").strip()
            if not page_content:
                continue

            header_parts: List[str] = []
            for key in ("h1", "h2", "h3", "h4"):
                value = str(metadata.get(key) or "").strip()
                if value:
                    header_parts.append(value)

            sections.append(
                {
                    "page_content": page_content,
                    "metadata": {
                        "header_path": _normalize_header_path(header_parts),
                    },
                }
            )

        return sections or _fallback_markdown_sections(rendered)
    except Exception:
        return _fallback_markdown_sections(rendered)


def _fitz_module() -> Optional[Any]:
    try:
        return importlib.import_module("fitz")
    except ImportError:
        return None


def _get_completion_response_callable() -> Optional[Callable[..., str]]:
    try:
        from GPT_handle import completion_response  # Local import to avoid hard dependency during tests.

        return completion_response
    except Exception:
        return None


def _call_lite_model_markdown(
    *,
    completion_callable: Optional[Callable[..., str]],
    model_id: str,
    page_number: int,
    image_data_url: str,
) -> str:
    if completion_callable is None or not model_id or not image_data_url:
        return ""

    user_prompt = (
        f"Convert this PDF page image to markdown. This is page {page_number}. "
        f"Keep the original reading order and preserve all visible table structures. "
        f"Return markdown only."
    )

    try:
        response = completion_callable(
            model=model_id,
            system_prompt=_LAYOUT_PARSE_SYSTEM_PROMPT.format(page=page_number),
            user_prompt=user_prompt,
            chat_history=None,
            temperature=0.0,
            image_urls=[image_data_url],
            pdf_inputs=None,
        )
        return _strip_markdown_fences(response)
    except Exception:
        return ""


def _call_lite_model_image_description(
    *,
    completion_callable: Optional[Callable[..., str]],
    model_id: str,
    page_number: int,
    image_index: int,
    image_data_url: str,
) -> str:
    if completion_callable is None or not model_id or not image_data_url:
        return ""

    user_prompt = (
        f"Analyze this document image from page {page_number}, figure {image_index}. "
        "Provide a technical description, key observed values/trends, and what it implies in context."
    )

    try:
        response = completion_callable(
            model=model_id,
            system_prompt=_IMAGE_DESCRIPTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            chat_history=None,
            temperature=0.1,
            image_urls=[image_data_url],
            pdf_inputs=None,
        )
        return _strip_markdown_fences(response)
    except Exception:
        return ""


def _extract_images_from_page(
    *,
    fitz_module: Any,
    pdf_doc: Any,
    page: Any,
    page_number: int,
    max_images_per_page: int,
    min_bytes: int,
) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []

    try:
        raw_images = page.get_images(full=True)
    except Exception:
        return images

    for index, raw in enumerate(raw_images, start=1):
        if len(images) >= max_images_per_page:
            break

        try:
            xref = int(raw[0])
        except Exception:
            continue

        try:
            pix = fitz_module.Pixmap(pdf_doc, xref)
            if pix.n > 4:
                pix = fitz_module.Pixmap(fitz_module.csRGB, pix)
            png_bytes = pix.tobytes("png")
        except Exception:
            continue

        if not png_bytes or len(png_bytes) < min_bytes:
            continue

        encoded = base64.b64encode(png_bytes).decode("utf-8")
        images.append(
            {
                "page": page_number,
                "image_index": index,
                "png_bytes": png_bytes,
                "data_url": f"data:image/png;base64,{encoded}",
            }
        )

    return images


def _render_page_image(
    *,
    page: Any,
    fitz_module: Any,
    dpi: int,
    max_dimension: int,
) -> Optional[Dict[str, Any]]:
    zoom = max(float(dpi), 72.0) / 72.0

    try:
        pix = page.get_pixmap(matrix=fitz_module.Matrix(zoom, zoom), alpha=False)
    except Exception:
        return None

    if max_dimension > 0:
        largest_side = max(pix.width, pix.height)
        if largest_side > max_dimension:
            scale = float(max_dimension) / float(largest_side)
            try:
                pix = page.get_pixmap(matrix=fitz_module.Matrix(zoom * scale, zoom * scale), alpha=False)
            except Exception:
                return None

    png_bytes = pix.tobytes("png")
    if not png_bytes:
        return None

    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return {
        "png_bytes": png_bytes,
        "data_url": f"data:image/png;base64,{encoded}",
        "width": pix.width,
        "height": pix.height,
    }


def _fallback_native_page_texts(filepath: str, max_chars: int) -> Tuple[List[Dict[str, Any]], int, str]:
    try:
        from pypdf import PdfReader
    except Exception:
        return [], 0, ""

    try:
        reader = PdfReader(filepath)
        page_count = len(reader.pages)
    except Exception:
        return [], 0, ""

    pages: List[Dict[str, Any]] = []
    total_chars = 0
    cap = max(0, int(max_chars))

    for index, pdf_page in enumerate(reader.pages, start=1):
        if cap and total_chars >= cap:
            break

        try:
            text = str(pdf_page.extract_text() or "").strip()
        except Exception:
            text = ""

        if not text:
            continue

        if cap:
            remaining = cap - total_chars
            if remaining <= 0:
                break
            text = text[:remaining]

        if not text:
            continue

        pages.append({"page": index, "text": text})
        total_chars += len(text)

    joined = "\n\n".join(item["text"] for item in pages).strip()
    return pages, page_count, joined


def _replace_placeholders_with_descriptions(
    *,
    markdown_text: str,
    page_number: int,
    images: Sequence[Dict[str, Any]],
    descriptions: Sequence[str],
) -> str:
    rendered = str(markdown_text or "").strip()
    if not rendered:
        rendered = ""

    placeholder_tokens = [f"[[IMAGE_P{page_number}_I{index}]]" for index in range(1, len(images) + 1)]

    for token in placeholder_tokens:
        if token not in rendered:
            if rendered:
                rendered += f"\n\n{token}"
            else:
                rendered = token

    for index, token in enumerate(placeholder_tokens, start=1):
        desc = ""
        if index - 1 < len(descriptions):
            desc = str(descriptions[index - 1] or "").strip()
        if not desc:
            desc = "No confident visual description was generated for this figure."

        block = f"### Image Description (Page {page_number}, Figure {index})\n{desc}"
        rendered = rendered.replace(token, block, 1)

    unused_descriptions = []
    for index, desc in enumerate(descriptions, start=1):
        if index > len(placeholder_tokens):
            text = str(desc or "").strip()
            if text:
                unused_descriptions.append((index, text))

    # Best-effort replacement for markdown image tags if emitted by the parser model.
    if unused_descriptions:
        def _replace_markdown_image(match: re.Match[str]) -> str:
            if not unused_descriptions:
                return match.group(0)
            idx, text = unused_descriptions.pop(0)
            return f"### Image Description (Page {page_number}, Figure {idx})\n{text}"

        rendered = _MARKDOWN_IMAGE_RE.sub(_replace_markdown_image, rendered)

    for index, text in unused_descriptions:
        rendered += f"\n\n### Image Description (Page {page_number}, Figure {index})\n{text}"

    return rendered.strip()


def _build_markdown_chunks(
    *,
    markdown_text: str,
    max_chunk_chars: int,
    max_items: int,
) -> List[Dict[str, Any]]:
    sections = _split_markdown_by_headers(markdown_text)
    if not sections:
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for section in sections:
        section_text = str(section.get("page_content") or "").strip()
        if not section_text:
            continue

        header_path = str((section.get("metadata") or {}).get("header_path") or "").strip()
        page_number = _extract_page_number_from_header_path(header_path)

        for piece in _chunk_text_without_breaking_atomic_blocks(section_text, max_chunk_chars):
            if len(chunks) >= max_items:
                return chunks

            content_type = _content_type_for_text(piece)
            chunk_counter += 1
            chunk = {
                "chunk_id": f"chunk-{chunk_counter:05d}",
                "text": piece,
                "page_start": page_number,
                "page_end": page_number,
                "page_number": page_number,
                "header_path": header_path,
                "content_type": content_type,
                "sources": ["layout_markdown"],
                "char_count": len(piece),
                "block_count": 1,
            }
            chunks.append(chunk)

    return chunks


def build_context_excerpt(chunks: Sequence[Dict[str, Any]], max_chars: int) -> str:
    if not chunks:
        return ""

    budget = max(500, int(max_chars))
    selected_parts: List[str] = []
    consumed = 0

    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue

        page_number = chunk.get("page_number")
        header_path = str(chunk.get("header_path") or "").strip()
        content_type = str(chunk.get("content_type") or "text").strip() or "text"

        if isinstance(page_number, int) and page_number > 0:
            prefix = f"According to page {page_number}"
        else:
            prefix = "According to the document"

        details = []
        if header_path:
            details.append(f"section {header_path}")
        if content_type:
            details.append(f"content type: {content_type}")

        if details:
            prefix += f" ({'; '.join(details)})"

        rendered = f"{prefix}:\n{text}"
        remaining = budget - consumed
        if remaining <= 0:
            break

        if len(rendered) > remaining:
            rendered = rendered[:remaining]

        if rendered:
            selected_parts.append(rendered)
            consumed += len(rendered)

        if consumed >= budget:
            break

    return "\n\n".join(selected_parts).strip()


def parse_pdf_document(
    filepath: str,
    settings: Dict[str, Any],
    log_error: Optional[Callable[[str, Exception], None]] = None,
) -> Dict[str, Any]:
    text_max_chars = _coerce_int(settings.get("pdf_text_max_chars", 500000), 500000, 1000, 5000000)
    chunk_max_chars = _coerce_int(settings.get("pdf_chunk_max_chars", 1800), 1800, 500, 12000)
    chunk_max_items = _coerce_int(settings.get("pdf_chunk_max_items", 400), 400, 1, 3000)

    layout_model = str(settings.get("pdf_layout_model") or "").strip()
    image_description_model = str(settings.get("pdf_image_description_model") or "").strip() or layout_model

    layout_dpi = _coerce_int(settings.get("pdf_layout_dpi", 180), 180, 72, 300)
    layout_max_dimension = _coerce_int(settings.get("pdf_layout_max_dimension", 2200), 2200, 512, 4096)
    layout_max_total_bytes = _coerce_int(
        settings.get("pdf_layout_max_total_bytes", 15728640),
        15728640,
        262144,
        104857600,
    )
    layout_max_pages = _coerce_int(settings.get("pdf_layout_max_pages", 200), 200, 1, 2000)

    image_desc_enabled = bool(settings.get("pdf_image_description_enabled", True))
    max_images_per_page = _coerce_int(settings.get("pdf_image_description_max_images_per_page", 4), 4, 0, 20)
    max_total_images = _coerce_int(settings.get("pdf_image_description_max_total_images", 120), 120, 0, 1000)
    min_image_bytes = _coerce_int(settings.get("pdf_image_min_bytes", 4096), 4096, 256, 5000000)

    completion_callable = _get_completion_response_callable()
    fitz_module = _fitz_module()

    if fitz_module is None:
        native_pages, page_count, native_text = _fallback_native_page_texts(filepath, text_max_chars)
        markdown_parts = [f"# Page {item['page']}\n\n{item['text']}" for item in native_pages]
        markdown_text = "\n\n".join(markdown_parts).strip()
        chunks = _build_markdown_chunks(
            markdown_text=markdown_text,
            max_chunk_chars=chunk_max_chars,
            max_items=chunk_max_items,
        )
        return {
            "page_count": page_count,
            "text": markdown_text[:text_max_chars],
            "native_text": native_text,
            "ocr_text": "",
            "has_extractable_text": bool(native_text),
            "visual_image_urls": [],
            "visual_meta": {
                "reason": "PyMuPDF is not installed; fallback to native text parsing",
                "rendered_pages": 0,
                "truncated": False,
            },
            "ocr_meta": {
                "enabled": False,
                "used": False,
                "reason": "OCR disabled in multimodal markdown pipeline",
            },
            "blocks": [{"page": item["page"], "text": item["text"], "source": "native_text"} for item in native_pages],
            "chunks": chunks,
            "parser_meta": {
                "mode": "fallback_native_text",
                "confidence": 0.45,
                "native_chars": len(native_text),
                "ocr_chars": 0,
                "block_count": len(native_pages),
                "chunk_count": len(chunks),
                "markdown_chars": len(markdown_text),
                "images_described": 0,
            },
        }

    try:
        pdf_doc = fitz_module.open(filepath)
    except Exception as exc:
        _log_error(log_error, f"pdf open failed for {filepath}", exc)
        return {
            "page_count": 0,
            "text": "",
            "native_text": "",
            "ocr_text": "",
            "has_extractable_text": False,
            "visual_image_urls": [],
            "visual_meta": {
                "reason": "failed to open PDF",
                "rendered_pages": 0,
                "truncated": False,
            },
            "ocr_meta": {
                "enabled": False,
                "used": False,
                "reason": "OCR disabled in multimodal markdown pipeline",
            },
            "blocks": [],
            "chunks": [],
            "parser_meta": {
                "mode": "empty",
                "confidence": 0.0,
                "native_chars": 0,
                "ocr_chars": 0,
                "block_count": 0,
                "chunk_count": 0,
                "markdown_chars": 0,
                "images_described": 0,
            },
        }

    page_count = len(pdf_doc)
    total_pages_to_parse = min(page_count, layout_max_pages)

    markdown_pages: List[Dict[str, Any]] = []
    rendered_page_images: List[str] = []
    blocks: List[Dict[str, Any]] = []
    native_text_parts: List[str] = []
    total_rendered_bytes = 0
    total_images_described = 0
    total_images_seen = 0
    truncated_by_bytes = False

    for page_index in range(total_pages_to_parse):
        page_number = page_index + 1
        try:
            page = pdf_doc.load_page(page_index)
        except Exception as exc:
            _log_error(log_error, f"pdf load page failed for page {page_number}", exc)
            continue

        try:
            page_native_text = str(page.get_text("text") or "").strip()
        except Exception:
            page_native_text = ""

        if page_native_text:
            native_text_parts.append(page_native_text)
            blocks.append({"page": page_number, "text": page_native_text, "source": "native_text"})

        rendered_page = _render_page_image(
            page=page,
            fitz_module=fitz_module,
            dpi=layout_dpi,
            max_dimension=layout_max_dimension,
        )

        if rendered_page is not None:
            png_bytes = rendered_page.get("png_bytes") or b""
            if total_rendered_bytes + len(png_bytes) <= layout_max_total_bytes:
                total_rendered_bytes += len(png_bytes)
                rendered_page_images.append(str(rendered_page.get("data_url") or ""))
            else:
                truncated_by_bytes = True
                rendered_page = None

        page_markdown = ""
        if rendered_page is not None and layout_model:
            page_markdown = _call_lite_model_markdown(
                completion_callable=completion_callable,
                model_id=layout_model,
                page_number=page_number,
                image_data_url=str(rendered_page.get("data_url") or ""),
            )

        if not page_markdown and page_native_text:
            page_markdown = page_native_text

        extracted_images: List[Dict[str, Any]] = []
        if max_images_per_page > 0 and total_images_seen < max_total_images:
            remaining_images_budget = max_total_images - total_images_seen
            page_image_cap = max(0, min(max_images_per_page, remaining_images_budget))
            extracted_images = _extract_images_from_page(
                fitz_module=fitz_module,
                pdf_doc=pdf_doc,
                page=page,
                page_number=page_number,
                max_images_per_page=page_image_cap,
                min_bytes=min_image_bytes,
            )

        total_images_seen += len(extracted_images)

        descriptions: List[str] = []
        if image_desc_enabled and extracted_images and image_description_model:
            for img in extracted_images:
                image_index = int(img.get("image_index") or 0)
                desc = _call_lite_model_image_description(
                    completion_callable=completion_callable,
                    model_id=image_description_model,
                    page_number=page_number,
                    image_index=image_index,
                    image_data_url=str(img.get("data_url") or ""),
                )
                descriptions.append(desc)
                if desc:
                    total_images_described += 1
        elif extracted_images:
            descriptions = [""] * len(extracted_images)

        page_markdown = _replace_placeholders_with_descriptions(
            markdown_text=page_markdown,
            page_number=page_number,
            images=extracted_images,
            descriptions=descriptions,
        )

        if not page_markdown:
            page_markdown = "[No extractable content on this page.]"

        markdown_pages.append(
            {
                "page": page_number,
                "markdown": page_markdown,
            }
        )

    pdf_doc.close()

    markdown_document_parts = []
    for item in markdown_pages:
        markdown_document_parts.append(f"# Page {item['page']}\n\n{item['markdown']}")

    markdown_document = "\n\n".join(markdown_document_parts).strip()
    if len(markdown_document) > text_max_chars:
        markdown_document = markdown_document[:text_max_chars]

    chunks = _build_markdown_chunks(
        markdown_text=markdown_document,
        max_chunk_chars=chunk_max_chars,
        max_items=chunk_max_items,
    )

    native_text = "\n\n".join(native_text_parts).strip()

    mode = "multimodal_layout_markdown"
    if not layout_model or completion_callable is None:
        mode = "native_markdown_fallback"

    confidence = 0.58
    if mode == "multimodal_layout_markdown":
        confidence += 0.2
    if total_images_described > 0:
        confidence += 0.14
    if chunks:
        confidence += 0.06
    confidence = max(0.0, min(confidence, 0.99))

    return {
        "page_count": page_count,
        "text": markdown_document,
        "native_text": native_text,
        "ocr_text": "",
        "has_extractable_text": bool(markdown_document.strip()),
        "visual_image_urls": [item for item in rendered_page_images if item],
        "visual_meta": {
            "reason": "",
            "rendered_pages": len(rendered_page_images),
            "truncated": truncated_by_bytes,
            "max_pages": total_pages_to_parse,
            "bytes": total_rendered_bytes,
        },
        "ocr_meta": {
            "enabled": False,
            "used": False,
            "reason": "OCR replaced by multimodal layout parsing",
        },
        "blocks": blocks,
        "chunks": chunks,
        "parser_meta": {
            "mode": mode,
            "confidence": confidence,
            "native_chars": len(native_text),
            "ocr_chars": 0,
            "block_count": len(blocks),
            "chunk_count": len(chunks),
            "markdown_chars": len(markdown_document),
            "images_detected": total_images_seen,
            "images_described": total_images_described,
        },
    }
