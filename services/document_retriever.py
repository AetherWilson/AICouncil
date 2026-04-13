import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-z0-9]+")
_EN_PAGE_RANGE_RE = re.compile(r"(?<![a-z0-9])pages?\s*(\d{1,4})\s*(?:-|~|to)\s*(\d{1,4})(?!\d)", re.IGNORECASE)
_EN_PAGE_SINGLE_RE = re.compile(r"(?<![a-z0-9])pages?\s*(\d{1,4})(?!\d)", re.IGNORECASE)
_CJK_PAGE_RANGE_RE = re.compile(r"第\s*(\d{1,4})\s*(?:到|至|-)\s*(\d{1,4})\s*[頁页]")
_CJK_PAGE_SINGLE_RE = re.compile(r"第\s*(\d{1,4})\s*[頁页]")
_CJK_PAGE_TAIL_RE = re.compile(r"(?<!\d)(\d{1,4})\s*[頁页]")

_VISUAL_QUERY_RE = re.compile(
    r"(figure|fig\.|chart|diagram|image|screenshot|table|plot|graph|layout|\bocr\b|圖片|圖表|圖像|插圖|示意圖|表格|流程圖|公式|版面)",
    re.IGNORECASE,
)
_SUMMARY_QUERY_RE = re.compile(
    r"(summary|summarize|overview|high[\s-]?level|tl;dr|總結|摘要|概覽|重點整理)",
    re.IGNORECASE,
)
_COMPARE_QUERY_RE = re.compile(
    r"(compare|comparison|contrast|difference|vs\.?|比較|差異|對照)",
    re.IGNORECASE,
)
_COMPLIANCE_QUERY_RE = re.compile(
    r"(full\s+document|entire\s+document|all\s+pages|every\s+page|exhaustive|compliance|audit|完整|全文|全部頁|逐頁|全書)",
    re.IGNORECASE,
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _tokenize(text: str) -> List[str]:
    rendered = str(text or "").lower().strip()
    if not rendered:
        return []

    tokens = _LATIN_RE.findall(rendered)
    cjk_chars = _CJK_RE.findall(rendered)
    tokens.extend(cjk_chars)
    return tokens


def _lexical_score(text: str, query_tokens: Sequence[str]) -> float:
    if not text or not query_tokens:
        return 0.0

    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0

    query_set = set(query_tokens)
    text_set = set(text_tokens)
    overlap = query_set.intersection(text_set)
    if not overlap:
        return 0.0

    coverage = float(len(overlap)) / float(max(1, len(query_set)))
    density = float(len(overlap)) / float(max(1, len(text_set)))
    return (0.75 * coverage) + (0.25 * density)


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    size = min(len(vec_a), len(vec_b))
    if size <= 0:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for index in range(size):
        a = float(vec_a[index])
        b = float(vec_b[index])
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _extract_embeddings(response: Any) -> List[List[float]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")

    vectors: List[List[float]] = []
    if not isinstance(data, list):
        return vectors

    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if isinstance(embedding, list):
            vectors.append([float(value) for value in embedding])

    return vectors


def _rank_by_default_order(chunks: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        chunks,
        key=lambda item: (
            int(item.get("page_start") or 0),
            int(item.get("page_end") or 0),
        ),
    )
    return list(ordered[:top_k])


def _apply_char_budget(chunks: Sequence[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    budget = max(500, int(max_chars))
    selected: List[Dict[str, Any]] = []
    consumed = 0

    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue

        length = len(text)
        if selected and (consumed + length) > budget:
            break

        selected.append(chunk)
        consumed += length

        if consumed >= budget:
            break

    if not selected and chunks:
        selected = [chunks[0]]

    return selected


def normalize_query_cache_key(query: str) -> str:
    rendered = str(query or "").strip().lower()
    if not rendered:
        return ""
    normalized = re.sub(r"\s+", " ", rendered)
    return normalized


def analyze_query_intent(query: str) -> Dict[str, Any]:
    rendered = str(query or "").strip()
    if not rendered:
        return {
            "intent": "default",
            "wants_visual": False,
            "wants_full_document": False,
            "page_hints": [],
        }

    page_hints = extract_page_hints(rendered, max_pages=12)
    wants_visual = bool(_VISUAL_QUERY_RE.search(rendered))
    wants_summary = bool(_SUMMARY_QUERY_RE.search(rendered))
    wants_compare = bool(_COMPARE_QUERY_RE.search(rendered))
    wants_full_doc = bool(_COMPLIANCE_QUERY_RE.search(rendered))

    intent = "default"
    if page_hints:
        intent = "page_lookup"
    elif wants_full_doc:
        intent = "compliance"
    elif wants_summary:
        intent = "summary"
    elif wants_compare:
        intent = "comparison"

    return {
        "intent": intent,
        "wants_visual": wants_visual,
        "wants_full_document": wants_full_doc,
        "page_hints": page_hints,
    }


def compute_retrieval_budget(
    *,
    intent: str,
    base_top_k: int,
    base_max_chars: int,
    context_max_chars: int,
) -> Dict[str, int]:
    top_k = max(1, min(int(base_top_k), 50))
    max_chars = max(500, int(base_max_chars))
    context_cap = max(500, int(context_max_chars))

    if intent == "page_lookup":
        top_k = max(2, min(top_k, 6))
        max_chars = min(context_cap, max(3000, int(max_chars * 0.8)))
    elif intent in {"summary", "comparison"}:
        top_k = min(24, max(top_k, 12))
        max_chars = min(context_cap, max(max_chars, 14000))
    elif intent == "compliance":
        top_k = min(32, max(top_k, 16))
        max_chars = min(context_cap, max(max_chars, 20000))

    return {
        "top_k": max(1, min(top_k, 50)),
        "max_chars": max(500, min(max_chars, context_cap)),
    }


def should_attach_visual_inputs(query: str, analyzed_intent: Optional[Dict[str, Any]] = None) -> bool:
    intent = analyzed_intent or analyze_query_intent(query)
    if bool(intent.get("wants_visual")):
        return True
    return False


def should_attach_pdf_binary(query: str, analyzed_intent: Optional[Dict[str, Any]] = None) -> bool:
    intent = analyzed_intent or analyze_query_intent(query)
    if bool(intent.get("wants_full_document")):
        return True
    if intent.get("intent") in {"summary", "compliance"}:
        return True
    return False


def _score_chunks(
    chunks: Sequence[Dict[str, Any]],
    query_text: str,
    embedding_client: Optional[Any],
    embedding_model: str,
    embedding_enabled: bool,
) -> Tuple[List[Tuple[float, int, Dict[str, Any]]], Dict[str, Any]]:
    if not chunks:
        return [], {"embedding_used": False, "embedding_failed": False}

    query_tokens = _tokenize(query_text)
    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "").strip()
        lexical = _lexical_score(text, query_tokens)
        scored.append((lexical, index, chunk))

    used_embedding = False
    embedding_failed = False
    if embedding_enabled and embedding_client is not None and str(embedding_model or "").strip() and query_text:
        try:
            inputs = [query_text]
            chunk_texts = [str(item.get("text") or "") for _, _, item in scored]
            inputs.extend(chunk_texts)

            response = embedding_client.embeddings.create(
                model=embedding_model,
                input=inputs,
            )
            vectors = _extract_embeddings(response)

            if len(vectors) == (1 + len(chunk_texts)):
                query_vec = vectors[0]
                combined: List[Tuple[float, int, Dict[str, Any]]] = []
                for idx, (lexical, original_index, chunk) in enumerate(scored):
                    emb = _cosine_similarity(query_vec, vectors[idx + 1])
                    score = (0.65 * emb) + (0.35 * lexical)
                    combined.append((score, original_index, chunk))
                scored = combined
                used_embedding = True
            else:
                embedding_failed = True
        except Exception:
            embedding_failed = True

    return scored, {
        "embedding_used": used_embedding,
        "embedding_failed": embedding_failed,
    }


def score_chunks_for_query(
    chunks: Sequence[Dict[str, Any]],
    query: str,
    embedding_client: Optional[Any] = None,
    embedding_model: str = "",
    embedding_enabled: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not chunks:
        return [], {"strategy": "empty", "embedding_used": False, "embedding_failed": False}

    query_text = str(query or "").strip()
    if not query_text:
        ordered = _rank_by_default_order(chunks, top_k=max(1, len(chunks)))
        scored = []
        for idx, chunk in enumerate(ordered):
            item = dict(chunk)
            item["_score"] = float(max(0.0, 1.0 - (idx * 0.01)))
            item["_rank"] = idx + 1
            item["_lexical_score"] = 0.0
            scored.append(item)
        return scored, {
            "strategy": "default_order",
            "embedding_used": False,
            "embedding_failed": False,
        }

    scored_tuples, emb_meta = _score_chunks(
        chunks,
        query_text=query_text,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        embedding_enabled=embedding_enabled,
    )
    scored_tuples.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    scored = []
    for idx, (score, _, chunk) in enumerate(scored_tuples):
        item = dict(chunk)
        item["_score"] = float(score)
        item["_rank"] = idx + 1
        item["_lexical_score"] = float(_lexical_score(str(chunk.get("text") or ""), _tokenize(query_text)))
        scored.append(item)

    strategy = "lexical"
    if emb_meta["embedding_used"]:
        strategy = "hybrid_embedding_lexical"
    elif emb_meta["embedding_failed"] and embedding_enabled:
        strategy = "lexical_embedding_fallback"

    return scored, {
        "strategy": strategy,
        "embedding_used": emb_meta["embedding_used"],
        "embedding_failed": emb_meta["embedding_failed"],
    }


def merge_page_and_semantic_candidates(
    *,
    query: str,
    scored_chunks: Sequence[Dict[str, Any]],
    page_focus_chunks: Sequence[Dict[str, Any]],
    page_hints: Sequence[int],
    top_k: int,
    max_chars: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalized_hints = sorted({int(p) for p in page_hints if int(p) > 0}) if page_hints else []
    dedup: Dict[str, Dict[str, Any]] = {}

    def dedup_key(chunk: Dict[str, Any]) -> str:
        page_start = int(chunk.get("page_start") or 0)
        page_end = int(chunk.get("page_end") or 0)
        text = str(chunk.get("text") or "")[:120].strip().lower()
        return f"{page_start}:{page_end}:{text}"

    for chunk in scored_chunks:
        item = dict(chunk)
        item["_score"] = float(item.get("_score") or 0.0)
        item["_from_page_focus"] = False
        key = dedup_key(item)
        existing = dedup.get(key)
        if existing is None or float(existing.get("_score") or 0.0) < item["_score"]:
            dedup[key] = item

    for chunk in page_focus_chunks:
        item = dict(chunk)
        base_score = float(item.get("_score") or 0.35)
        page_boost = 0.18
        if normalized_hints:
            start = int(item.get("page_start") or 0)
            end = int(item.get("page_end") or start)
            if any(start <= hint <= end for hint in normalized_hints):
                page_boost = 0.28
        item["_score"] = base_score + page_boost
        item["_from_page_focus"] = True
        key = dedup_key(item)
        existing = dedup.get(key)
        if existing is None or float(existing.get("_score") or 0.0) < item["_score"]:
            dedup[key] = item

    merged = list(dedup.values())
    merged.sort(
        key=lambda item: (
            float(item.get("_score") or 0.0),
            -int(item.get("page_start") or 0),
        ),
        reverse=True,
    )

    capped_top_k = max(1, min(int(top_k), 50))
    selected_pool = merged[:capped_top_k]
    selected = _apply_char_budget(selected_pool, max_chars=max_chars)

    top_score = float(selected_pool[0].get("_score") or 0.0) if selected_pool else 0.0
    second_score = float(selected_pool[1].get("_score") or 0.0) if len(selected_pool) > 1 else 0.0
    score_gap = max(0.0, top_score - second_score)
    confidence = _clamp((0.7 * top_score) + (0.3 * score_gap), 0.0, 1.0)

    return selected, {
        "strategy": "hybrid_page_semantic_lexical",
        "selected": len(selected),
        "total_candidates": len(merged),
        "page_focus_candidates": len(page_focus_chunks),
        "semantic_candidates": len(scored_chunks),
        "confidence": confidence,
        "top_score": top_score,
    }


def rank_chunks_for_query(
    chunks: Sequence[Dict[str, Any]],
    query: str,
    top_k: int = 8,
    max_chars: int = 12000,
    embedding_client: Optional[Any] = None,
    embedding_model: str = "",
    embedding_enabled: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not chunks:
        return [], {"strategy": "empty", "selected": 0, "total": 0}

    capped_top_k = max(1, min(int(top_k), 50))
    query_text = str(query or "").strip()
    if not query_text:
        selected = _apply_char_budget(_rank_by_default_order(chunks, capped_top_k), max_chars)
        return selected, {
            "strategy": "default_order",
            "selected": len(selected),
            "total": len(chunks),
        }

    scored_chunks, scored_meta = score_chunks_for_query(
        chunks,
        query=query_text,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        embedding_enabled=embedding_enabled,
    )

    top_ranked = scored_chunks[:capped_top_k]
    selected = _apply_char_budget(top_ranked, max_chars)

    strategy = str(scored_meta.get("strategy") or "lexical")

    return selected, {
        "strategy": strategy,
        "selected": len(selected),
        "total": len(chunks),
        "embedding_used": bool(scored_meta.get("embedding_used", False)),
        "embedding_failed": bool(scored_meta.get("embedding_failed", False)),
    }


def extract_page_hints(query: str, max_pages: int = 8, max_range_width: int = 20) -> List[int]:
    rendered = str(query or "").strip()
    if not rendered:
        return []

    page_numbers: List[int] = []

    for match in _EN_PAGE_RANGE_RE.finditer(rendered):
        start = int(match.group(1))
        end = int(match.group(2))
        low = min(start, end)
        high = max(start, end)
        if (high - low + 1) > max_range_width:
            high = low + max_range_width - 1
        for page in range(low, high + 1):
            if page > 0:
                page_numbers.append(page)

    for match in _CJK_PAGE_RANGE_RE.finditer(rendered):
        start = int(match.group(1))
        end = int(match.group(2))
        low = min(start, end)
        high = max(start, end)
        if (high - low + 1) > max_range_width:
            high = low + max_range_width - 1
        for page in range(low, high + 1):
            if page > 0:
                page_numbers.append(page)

    for match in _EN_PAGE_SINGLE_RE.finditer(rendered):
        page = int(match.group(1))
        if page > 0:
            page_numbers.append(page)

    for match in _CJK_PAGE_SINGLE_RE.finditer(rendered):
        page = int(match.group(1))
        if page > 0:
            page_numbers.append(page)

    for match in _CJK_PAGE_TAIL_RE.finditer(rendered):
        page = int(match.group(1))
        if page > 0:
            page_numbers.append(page)

    deduped = sorted(set(page_numbers))
    if max_pages <= 0:
        return deduped
    return deduped[:max_pages]
