import base64
import importlib
import io
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _log_error(log_error: Optional[Callable[[str, Exception], None]], context: str, exc: Exception) -> None:
    if callable(log_error):
        log_error(context, exc)


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(parsed, maximum))


def _normalize_ocr_languages(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        langs = [str(item).strip() for item in value if str(item).strip()]
        return langs or ["en"]

    if isinstance(value, str):
        langs = [part.strip() for part in value.split(",") if part.strip()]
        return langs or ["en"]

    return ["en"]


def _extract_native_page_texts(
    filepath: str,
    max_chars: int,
    log_error: Optional[Callable[[str, Exception], None]] = None,
) -> Tuple[List[Dict[str, Any]], int, str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return [], 0, ""

    try:
        reader = PdfReader(filepath)
        page_count = len(reader.pages)
    except Exception as exc:
        _log_error(log_error, f"native pdf parse failed for {filepath}", exc)
        return [], 0, ""

    page_texts: List[Dict[str, Any]] = []
    total = 0
    hard_max = max(0, int(max_chars))

    for index, page in enumerate(reader.pages):
        if hard_max and total >= hard_max:
            break

        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""

        if not text:
            continue

        if hard_max:
            remaining = hard_max - total
            if remaining <= 0:
                break
            snippet = text[:remaining]
        else:
            snippet = text

        if not snippet:
            continue

        page_texts.append(
            {
                "page": index + 1,
                "text": snippet,
                "source": "native_text",
                "confidence": 0.95,
            }
        )
        total += len(snippet)

    native_text = "\n\n".join(item["text"] for item in page_texts).strip()
    return page_texts, page_count, native_text


def _render_visual_pages(
    filepath: str,
    max_pages: int,
    dpi: int,
    max_total_bytes: int,
    max_dimension: int,
    log_error: Optional[Callable[[str, Exception], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_pages <= 0:
        return [], {"reason": "pdf visual rendering disabled by page limit", "rendered_pages": 0, "truncated": False}

    try:
        fitz = importlib.import_module("fitz")
    except ImportError:
        return [], {"reason": "PyMuPDF is not installed", "rendered_pages": 0, "truncated": False}

    try:
        with fitz.open(filepath) as pdf_doc:
            page_count = len(pdf_doc)
            page_limit = min(max_pages, page_count)
            zoom = max(float(dpi), 72.0) / 72.0

            total_bytes = 0
            truncated = False
            stop_reason = ""
            rendered_pages: List[Dict[str, Any]] = []

            for page_index in range(page_limit):
                page = pdf_doc.load_page(page_index)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

                if max_dimension > 0:
                    largest_side = max(pixmap.width, pixmap.height)
                    if largest_side > max_dimension:
                        scale = float(max_dimension) / float(largest_side)
                        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom * scale, zoom * scale), alpha=False)

                png_bytes = pixmap.tobytes("png")
                if not png_bytes:
                    continue

                if total_bytes + len(png_bytes) > max_total_bytes:
                    truncated = True
                    stop_reason = "stopped at configured PDF visual size limit"
                    break

                total_bytes += len(png_bytes)
                encoded = base64.b64encode(png_bytes).decode("utf-8")

                rendered_pages.append(
                    {
                        "page": page_index + 1,
                        "width": pixmap.width,
                        "height": pixmap.height,
                        "png_bytes": png_bytes,
                        "data_url": f"data:image/png;base64,{encoded}",
                    }
                )

            meta = {
                "page_count": page_count,
                "rendered_pages": len(rendered_pages),
                "truncated": truncated,
                "reason": stop_reason,
            }
            if not rendered_pages and not stop_reason:
                meta["reason"] = "no renderable PDF pages found"

            return rendered_pages, meta
    except Exception as exc:
        _log_error(log_error, f"pdf visual render failed for {filepath}", exc)
        return [], {"reason": "pdf page rendering failed", "rendered_pages": 0, "truncated": False}


def _run_ocr_on_pages(
    rendered_pages: Sequence[Dict[str, Any]],
    enabled: bool,
    languages: Sequence[str],
    max_chars: int,
    log_error: Optional[Callable[[str, Exception], None]] = None,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    if not enabled:
        return [], "", {"enabled": False, "used": False, "reason": "ocr disabled"}

    if not rendered_pages:
        return [], "", {"enabled": True, "used": False, "reason": "no rendered pages available"}

    try:
        easyocr = importlib.import_module("easyocr")
    except ImportError:
        return [], "", {"enabled": True, "used": False, "reason": "easyocr not installed"}

    try:
        reader = easyocr.Reader(list(languages), gpu=False, verbose=False)
    except TypeError:
        reader = easyocr.Reader(list(languages), gpu=False)
    except Exception as exc:
        _log_error(log_error, "easyocr reader initialization failed", exc)
        return [], "", {"enabled": True, "used": False, "reason": "ocr initialization failed"}

    ocr_blocks: List[Dict[str, Any]] = []
    total_chars = 0
    hard_max = max(0, int(max_chars))

    for page_info in rendered_pages:
        if hard_max and total_chars >= hard_max:
            break

        image_bytes = page_info.get("png_bytes")
        if not image_bytes:
            continue

        page_text = ""
        try:
            lines = reader.readtext(image_bytes, detail=0, paragraph=True)
            page_text = "\n".join(str(item).strip() for item in lines if str(item).strip())
        except Exception:
            try:
                import numpy as np
                from PIL import Image

                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                lines = reader.readtext(np.array(image), detail=0, paragraph=True)
                page_text = "\n".join(str(item).strip() for item in lines if str(item).strip())
            except Exception as exc:
                _log_error(log_error, f"ocr failed on page {page_info.get('page')}", exc)
                continue

        page_text = page_text.strip()
        if not page_text:
            continue

        if hard_max:
            remaining = hard_max - total_chars
            if remaining <= 0:
                break
            page_text = page_text[:remaining]

        if not page_text:
            continue

        ocr_blocks.append(
            {
                "page": int(page_info.get("page") or 0),
                "text": page_text,
                "source": "ocr_text",
                "confidence": 0.62,
            }
        )
        total_chars += len(page_text)

    ocr_text = "\n\n".join(item["text"] for item in ocr_blocks).strip()
    meta = {
        "enabled": True,
        "used": bool(ocr_blocks),
        "pages_processed": len(rendered_pages),
        "pages_with_text": len(ocr_blocks),
        "languages": list(languages),
        "chars": len(ocr_text),
        "reason": "" if ocr_blocks else "ocr produced no text",
    }
    return ocr_blocks, ocr_text, meta


def _build_chunks_from_blocks(
    blocks: Sequence[Dict[str, Any]],
    chunk_max_chars: int,
    chunk_max_items: int,
) -> List[Dict[str, Any]]:
    if not blocks:
        return []

    max_chars = max(300, int(chunk_max_chars))
    max_items = max(1, int(chunk_max_items))

    ordered = sorted(blocks, key=lambda item: (int(item.get("page") or 0), str(item.get("source") or "")))

    chunks: List[Dict[str, Any]] = []
    current_texts: List[str] = []
    current_sources: List[str] = []
    current_pages: List[int] = []
    current_chars = 0

    def flush_chunk() -> None:
        nonlocal current_texts, current_sources, current_pages, current_chars
        if not current_texts:
            return

        unique_sources = []
        for source in current_sources:
            if source and source not in unique_sources:
                unique_sources.append(source)

        chunk_text = "\n\n".join(current_texts).strip()
        if not chunk_text:
            current_texts, current_sources, current_pages, current_chars = [], [], [], 0
            return

        chunks.append(
            {
                "text": chunk_text,
                "page_start": min(current_pages) if current_pages else None,
                "page_end": max(current_pages) if current_pages else None,
                "sources": unique_sources,
                "char_count": len(chunk_text),
                "block_count": len(current_texts),
            }
        )
        current_texts, current_sources, current_pages, current_chars = [], [], [], 0

    for block in ordered:
        text = str(block.get("text") or "").strip()
        if not text:
            continue

        text_len = len(text)
        if current_texts and (current_chars + text_len > max_chars):
            flush_chunk()
            if len(chunks) >= max_items:
                break

        if len(chunks) >= max_items:
            break

        if text_len > max_chars:
            text = text[:max_chars]
            text_len = len(text)

        current_texts.append(text)
        current_sources.append(str(block.get("source") or ""))
        current_pages.append(int(block.get("page") or 0))
        current_chars += text_len

    if len(chunks) < max_items:
        flush_chunk()

    return chunks[:max_items]


def build_context_excerpt(chunks: Sequence[Dict[str, Any]], max_chars: int) -> str:
    if not chunks:
        return ""

    hard_max = max(500, int(max_chars))
    output_parts: List[str] = []
    total = 0

    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue

        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if isinstance(page_start, int) and isinstance(page_end, int):
            if page_start == page_end:
                prefix = f"[Page {page_start}] "
            else:
                prefix = f"[Pages {page_start}-{page_end}] "
        else:
            prefix = ""

        rendered = f"{prefix}{text}".strip()
        remaining = hard_max - total
        if remaining <= 0:
            break

        if len(rendered) > remaining:
            rendered = rendered[:remaining]

        if rendered:
            output_parts.append(rendered)
            total += len(rendered)

        if total >= hard_max:
            break

    return "\n\n".join(output_parts).strip()


def parse_pdf_document(
    filepath: str,
    settings: Dict[str, Any],
    log_error: Optional[Callable[[str, Exception], None]] = None,
) -> Dict[str, Any]:
    text_max_chars = _coerce_int(settings.get("pdf_text_max_chars", 20000), 20000, 1000, 100000)
    visual_enabled = bool(settings.get("pdf_visual_enabled", True))
    visual_max_pages = _coerce_int(settings.get("pdf_visual_max_pages", 3), 3, 0, 12)
    visual_dpi = _coerce_int(settings.get("pdf_visual_dpi", 150), 150, 72, 300)
    visual_max_total_bytes = _coerce_int(settings.get("pdf_visual_max_total_bytes", 6291456), 6291456, 262144, 20971520)
    visual_max_dimension = _coerce_int(settings.get("pdf_visual_max_dimension", 2048), 2048, 512, 4096)

    ocr_enabled = bool(settings.get("pdf_ocr_enabled", True))
    ocr_min_text_chars = _coerce_int(settings.get("pdf_ocr_min_text_chars", 1200), 1200, 0, 50000)
    ocr_max_chars = _coerce_int(settings.get("pdf_ocr_max_chars", 12000), 12000, 500, 80000)
    ocr_languages = _normalize_ocr_languages(settings.get("pdf_ocr_languages", ["en"]))

    chunk_max_chars = _coerce_int(settings.get("pdf_chunk_max_chars", 1200), 1200, 300, 5000)
    chunk_max_items = _coerce_int(settings.get("pdf_chunk_max_items", 20), 20, 1, 200)

    native_blocks, page_count, native_text = _extract_native_page_texts(
        filepath,
        max_chars=text_max_chars,
        log_error=log_error,
    )

    rendered_pages: List[Dict[str, Any]] = []
    visual_meta: Dict[str, Any]
    if visual_enabled:
        rendered_pages, visual_meta = _render_visual_pages(
            filepath,
            max_pages=visual_max_pages,
            dpi=visual_dpi,
            max_total_bytes=visual_max_total_bytes,
            max_dimension=visual_max_dimension,
            log_error=log_error,
        )
    else:
        visual_meta = {"reason": "pdf visual rendering disabled in config", "rendered_pages": 0, "truncated": False}

    if page_count <= 0 and isinstance(visual_meta.get("page_count"), int):
        page_count = int(visual_meta.get("page_count") or 0)

    should_run_ocr = bool(ocr_enabled and len(native_text) < ocr_min_text_chars and rendered_pages)
    ocr_blocks, ocr_text, ocr_meta = _run_ocr_on_pages(
        rendered_pages,
        enabled=should_run_ocr,
        languages=ocr_languages,
        max_chars=ocr_max_chars,
        log_error=log_error,
    )

    blocks: List[Dict[str, Any]] = []
    blocks.extend(native_blocks)
    blocks.extend(ocr_blocks)
    chunks = _build_chunks_from_blocks(blocks, chunk_max_chars=chunk_max_chars, chunk_max_items=chunk_max_items)

    merged_text_parts = []
    if native_text:
        merged_text_parts.append(native_text)
    if ocr_text:
        if native_text:
            merged_text_parts.append("[OCR supplement]")
        merged_text_parts.append(ocr_text)
    merged_text = "\n\n".join(part for part in merged_text_parts if part).strip()

    if native_text and ocr_meta.get("used"):
        mode = "hybrid_native_plus_ocr"
    elif native_text and rendered_pages:
        mode = "native_plus_visual"
    elif native_text:
        mode = "native_text_only"
    elif ocr_meta.get("used"):
        mode = "ocr_only"
    elif rendered_pages:
        mode = "visual_only"
    else:
        mode = "empty"

    confidence = 0.35
    if native_text:
        confidence += 0.4
    if ocr_meta.get("used"):
        confidence += 0.18
    if rendered_pages:
        confidence += 0.07
    confidence = max(0.0, min(0.99, confidence))

    parser_meta = {
        "mode": mode,
        "confidence": confidence,
        "native_chars": len(native_text),
        "ocr_chars": len(ocr_text),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
    }

    visual_image_urls = [item.get("data_url") for item in rendered_pages if item.get("data_url")]

    return {
        "page_count": page_count,
        "text": merged_text,
        "native_text": native_text,
        "ocr_text": ocr_text,
        "has_extractable_text": bool(native_text),
        "visual_image_urls": visual_image_urls,
        "visual_meta": visual_meta,
        "ocr_meta": ocr_meta,
        "blocks": blocks,
        "chunks": chunks,
        "parser_meta": parser_meta,
    }
