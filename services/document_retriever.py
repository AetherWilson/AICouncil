import hashlib
import math
import os
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-z0-9]+")
_EN_PAGE_RANGE_RE = re.compile(r"(?<![a-z0-9])pages?\s*(\d{1,4})\s*(?:-|~|to)\s*(\d{1,4})(?!\d)", re.IGNORECASE)
_EN_PAGE_SINGLE_RE = re.compile(r"(?<![a-z0-9])pages?\s*(\d{1,4})(?!\d)", re.IGNORECASE)
_CJK_PAGE_RANGE_RE = re.compile(r"第\s*(\d{1,4})\s*(?:到|至|-)\s*(\d{1,4})\s*[頁页]")
_CJK_PAGE_SINGLE_RE = re.compile(r"第\s*(\d{1,4})\s*[頁页]")
_CJK_PAGE_TAIL_RE = re.compile(r"(?<!\d)(\d{1,4})\s*[頁页]")

_VISUAL_QUERY_RE = re.compile(
    r"(figure|fig\.|chart|diagram|image|screenshot|table|plot|graph|layout|ocr|圖片|圖表|圖像|插圖|示意圖|表格|流程圖|公式|版面)",
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

_INDEX_CACHE_LOCK = threading.Lock()
_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _tokenize(text: str) -> List[str]:
    rendered = str(text or "").lower().strip()
    if not rendered:
        return []

    tokens = _LATIN_RE.findall(rendered)
    tokens.extend(_CJK_RE.findall(rendered))
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


def _embed_texts(
    embedding_client: Optional[Any],
    embedding_model: str,
    texts: Sequence[str],
) -> List[List[float]]:
    if embedding_client is None or not str(embedding_model or "").strip() or not texts:
        return []

    response = embedding_client.embeddings.create(model=embedding_model, input=list(texts))
    return _extract_embeddings(response)


def _sanitize_collection_name(value: str) -> str:
    rendered = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip().lower())
    rendered = rendered.strip("_")
    if not rendered:
        rendered = "pdf_doc"
    if len(rendered) > 50:
        rendered = rendered[:50]
    return rendered


def _get_chroma_client(persist_directory: str) -> Optional[Any]:
    if not persist_directory:
        return None

    try:
        import chromadb

        os.makedirs(persist_directory, exist_ok=True)
        return chromadb.PersistentClient(path=persist_directory)
    except Exception:
        return None


def _document_signature(chunks: Sequence[Dict[str, Any]]) -> str:
    hasher = hashlib.sha1()
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        page = str(chunk.get("page_number") or chunk.get("page_start") or "")
        header = str(chunk.get("header_path") or "")
        content_type = str(chunk.get("content_type") or "")
        hasher.update(page.encode("utf-8"))
        hasher.update(header.encode("utf-8"))
        hasher.update(content_type.encode("utf-8"))
        hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def _build_bm25_state(chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ids: List[str] = []
    texts: List[str] = []
    corpus_tokens: List[List[str]] = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk.get("chunk_id") or f"chunk-{index:05d}")
        text = str(chunk.get("text") or "").strip()
        ids.append(chunk_id)
        texts.append(text)
        corpus_tokens.append(_tokenize(text))

    bm25 = None
    if BM25Okapi is not None and corpus_tokens:
        try:
            bm25 = BM25Okapi(corpus_tokens)
        except Exception:
            bm25 = None

    return {
        "ids": ids,
        "texts": texts,
        "tokens": corpus_tokens,
        "bm25": bm25,
    }


def ensure_document_index(
    *,
    document_id: str,
    chunks: Sequence[Dict[str, Any]],
    embedding_client: Optional[Any],
    embedding_model: str,
    persist_directory: str,
) -> Dict[str, Any]:
    normalized_document_id = str(document_id or "").strip() or "pdf_doc"
    signature = _document_signature(chunks)

    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(normalized_document_id)
        if isinstance(cached, dict):
            if (
                cached.get("signature") == signature
                and cached.get("embedding_model") == str(embedding_model or "").strip()
                and cached.get("persist_directory") == str(persist_directory or "").strip()
            ):
                return cached

    chunk_list = [dict(item) for item in chunks]
    chunk_map: Dict[str, Dict[str, Any]] = {}
    chunk_ids: List[str] = []
    chunk_texts: List[str] = []

    for index, chunk in enumerate(chunk_list, start=1):
        chunk_id = str(chunk.get("chunk_id") or f"chunk-{index:05d}")
        chunk["chunk_id"] = chunk_id
        chunk_ids.append(chunk_id)
        chunk_texts.append(str(chunk.get("text") or ""))
        chunk_map[chunk_id] = chunk

    bm25_state = _build_bm25_state(chunk_list)

    collection_name = _sanitize_collection_name(f"pdf_{normalized_document_id}")
    collection = None
    vector_index_ready = False
    vector_index_error = ""

    client = _get_chroma_client(str(persist_directory or "").strip())
    if client is not None and chunk_ids:
        try:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            embeddings = _embed_texts(embedding_client, embedding_model, chunk_texts)
            if len(embeddings) != len(chunk_texts):
                embeddings = []

            metadatas = []
            for chunk in chunk_list:
                page_number = chunk.get("page_number")
                if not isinstance(page_number, int):
                    page_number = chunk.get("page_start")
                if not isinstance(page_number, int):
                    page_number = 0

                metadatas.append(
                    {
                        "page_number": int(page_number),
                        "header_path": str(chunk.get("header_path") or ""),
                        "content_type": str(chunk.get("content_type") or "text"),
                        "chunk_id": str(chunk.get("chunk_id") or ""),
                    }
                )

            add_payload: Dict[str, Any] = {
                "ids": chunk_ids,
                "documents": chunk_texts,
                "metadatas": metadatas,
            }
            if embeddings:
                add_payload["embeddings"] = embeddings

            collection.add(**add_payload)
            vector_index_ready = bool(embeddings)
        except Exception as exc:
            collection = None
            vector_index_error = str(exc)

    state = {
        "document_id": normalized_document_id,
        "signature": signature,
        "embedding_model": str(embedding_model or "").strip(),
        "persist_directory": str(persist_directory or "").strip(),
        "collection_name": collection_name,
        "collection": collection,
        "chunk_map": chunk_map,
        "chunks": chunk_list,
        "chunk_ids": chunk_ids,
        "bm25_state": bm25_state,
        "vector_index_ready": vector_index_ready,
        "vector_index_error": vector_index_error,
    }

    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE[normalized_document_id] = state

    return state


def _vector_rank(
    *,
    state: Dict[str, Any],
    query: str,
    embedding_client: Optional[Any],
    embedding_model: str,
    top_n: int,
) -> Tuple[List[Tuple[str, float]], bool, bool]:
    if not query:
        return [], False, False

    collection = state.get("collection")
    if collection is None:
        return [], False, False

    try:
        query_vecs = _embed_texts(embedding_client, embedding_model, [query])
        if len(query_vecs) != 1:
            return [], False, True

        response = collection.query(
            query_embeddings=[query_vecs[0]],
            n_results=max(1, int(top_n)),
            include=["distances", "metadatas"],
        )
        ids = []
        distances = []
        if isinstance(response, dict):
            ids = (response.get("ids") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]

        ranked: List[Tuple[str, float]] = []
        for index, chunk_id in enumerate(ids):
            if not chunk_id:
                continue
            distance = 0.0
            if index < len(distances):
                try:
                    distance = float(distances[index] or 0.0)
                except (TypeError, ValueError):
                    distance = 0.0
            score = max(0.0, 1.0 - distance)
            ranked.append((str(chunk_id), score))

        return ranked, True, False
    except Exception:
        return [], False, True


def _bm25_rank(
    *,
    state: Dict[str, Any],
    query: str,
    top_n: int,
) -> List[Tuple[str, float]]:
    bm25_state = state.get("bm25_state") if isinstance(state, dict) else {}
    bm25 = bm25_state.get("bm25") if isinstance(bm25_state, dict) else None
    if bm25 is None:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    try:
        scores = bm25.get_scores(query_tokens)
    except Exception:
        return []

    ranked = []
    ids = list(bm25_state.get("ids") or [])
    for index, value in enumerate(scores):
        if index >= len(ids):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        ranked.append((ids[index], score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[: max(1, int(top_n))]


def _fuse_rankings(
    *,
    vector_ranked: Sequence[Tuple[str, float]],
    bm25_ranked: Sequence[Tuple[str, float]],
    chunks: Dict[str, Dict[str, Any]],
    top_k: int,
    max_chars: int,
    query: str,
    vector_weight: float,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    rrf_k = 60.0

    vector_positions = {chunk_id: idx + 1 for idx, (chunk_id, _) in enumerate(vector_ranked)}
    bm25_positions = {chunk_id: idx + 1 for idx, (chunk_id, _) in enumerate(bm25_ranked)}

    vector_scores = {chunk_id: score for chunk_id, score in vector_ranked}
    bm25_scores = {chunk_id: score for chunk_id, score in bm25_ranked}

    candidate_ids = set(vector_positions.keys()).union(set(bm25_positions.keys()))
    if not candidate_ids:
        candidate_ids = set(chunks.keys())

    fused: List[Tuple[str, float]] = []
    query_tokens = _tokenize(query)

    for chunk_id in candidate_ids:
        chunk = chunks.get(chunk_id)
        if not chunk:
            continue

        vector_component = 0.0
        if chunk_id in vector_positions:
            vector_component = vector_weight * (1.0 / (rrf_k + float(vector_positions[chunk_id])))

        bm25_component = 0.0
        if chunk_id in bm25_positions:
            bm25_component = bm25_weight * (1.0 / (rrf_k + float(bm25_positions[chunk_id])))

        lexical_component = 0.15 * _lexical_score(str(chunk.get("text") or ""), query_tokens)
        fused_score = vector_component + bm25_component + lexical_component
        fused.append((chunk_id, fused_score))

    fused.sort(key=lambda item: item[1], reverse=True)

    selected: List[Dict[str, Any]] = []
    consumed = 0
    cap = max(500, int(max_chars))

    for rank, (chunk_id, score) in enumerate(fused, start=1):
        if len(selected) >= max(1, int(top_k)):
            break

        chunk = dict(chunks.get(chunk_id) or {})
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue

        if selected and consumed + len(text) > cap:
            break

        chunk["_score"] = float(score)
        chunk["_rank"] = rank
        chunk["_vector_rank"] = vector_positions.get(chunk_id)
        chunk["_bm25_rank"] = bm25_positions.get(chunk_id)
        chunk["_vector_raw_score"] = float(vector_scores.get(chunk_id) or 0.0)
        chunk["_bm25_raw_score"] = float(bm25_scores.get(chunk_id) or 0.0)

        selected.append(chunk)
        consumed += len(text)

    if not selected and fused:
        top_chunk_id = fused[0][0]
        top_chunk = dict(chunks.get(top_chunk_id) or {})
        top_chunk["_score"] = float(fused[0][1])
        top_chunk["_rank"] = 1
        selected = [top_chunk]

    return selected


def retrieve_hybrid_chunks(
    *,
    document_id: str,
    chunks: Sequence[Dict[str, Any]],
    query: str,
    embedding_client: Optional[Any],
    embedding_model: str,
    persist_directory: str,
    top_k: int,
    max_chars: int,
    candidate_pool: int,
    vector_weight: float = 0.55,
    bm25_weight: float = 0.45,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not chunks:
        return [], {"strategy": "empty", "selected": 0, "total": 0}

    query_text = str(query or "").strip()
    if not query_text:
        ordered = sorted(
            [dict(item) for item in chunks],
            key=lambda item: (
                int(item.get("page_number") or item.get("page_start") or 0),
                str(item.get("header_path") or ""),
            ),
        )
        selected = []
        consumed = 0
        cap = max(500, int(max_chars))
        for index, chunk in enumerate(ordered, start=1):
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            if selected and consumed + len(text) > cap:
                break
            chunk["_score"] = float(max(0.0, 1.0 - (index * 0.01)))
            chunk["_rank"] = index
            selected.append(chunk)
            consumed += len(text)
            if len(selected) >= max(1, int(top_k)):
                break

        return selected, {
            "strategy": "default_order",
            "selected": len(selected),
            "total": len(chunks),
            "vector_used": False,
            "vector_failed": False,
            "bm25_used": bool(BM25Okapi is not None),
            "confidence": 0.45,
        }

    state = ensure_document_index(
        document_id=document_id,
        chunks=chunks,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        persist_directory=persist_directory,
    )

    top_n = max(max(1, int(top_k)), max(1, int(candidate_pool)))
    vector_ranked, vector_used, vector_failed = _vector_rank(
        state=state,
        query=query_text,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
        top_n=top_n,
    )
    bm25_ranked = _bm25_rank(state=state, query=query_text, top_n=top_n)

    chunk_map = state.get("chunk_map") if isinstance(state.get("chunk_map"), dict) else {}
    selected = _fuse_rankings(
        vector_ranked=vector_ranked,
        bm25_ranked=bm25_ranked,
        chunks=chunk_map,
        top_k=max(1, int(top_k)),
        max_chars=max_chars,
        query=query_text,
        vector_weight=_clamp(vector_weight, 0.0, 1.0),
        bm25_weight=_clamp(bm25_weight, 0.0, 1.0),
    )

    top_score = float(selected[0].get("_score") or 0.0) if selected else 0.0
    second_score = float(selected[1].get("_score") or 0.0) if len(selected) > 1 else 0.0
    gap = max(0.0, top_score - second_score)
    confidence = _clamp((0.68 * top_score) + (0.32 * gap), 0.0, 1.0)

    return selected, {
        "strategy": "hybrid_vector_bm25",
        "selected": len(selected),
        "total": len(chunks),
        "vector_used": vector_used,
        "vector_failed": vector_failed,
        "bm25_used": bool(bm25_ranked),
        "vector_candidates": len(vector_ranked),
        "bm25_candidates": len(bm25_ranked),
        "confidence": confidence,
        "top_score": top_score,
        "index_ready": bool(state.get("vector_index_ready", False)),
        "index_error": str(state.get("vector_index_error") or ""),
    }


def normalize_query_cache_key(query: str) -> str:
    rendered = str(query or "").strip().lower()
    if not rendered:
        return ""
    return re.sub(r"\s+", " ", rendered)


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
    top_k = max(1, min(int(base_top_k), 80))
    max_chars = max(500, int(base_max_chars))
    context_cap = max(500, int(context_max_chars))

    if intent == "page_lookup":
        top_k = max(3, min(top_k, 10))
        max_chars = min(context_cap, max(3500, int(max_chars * 0.75)))
    elif intent in {"summary", "comparison"}:
        top_k = min(40, max(top_k, 16))
        max_chars = min(context_cap, max(max_chars, 18000))
    elif intent == "compliance":
        top_k = min(60, max(top_k, 24))
        max_chars = min(context_cap, max(max_chars, 32000))

    return {
        "top_k": max(1, min(top_k, 80)),
        "max_chars": max(500, min(max_chars, context_cap)),
    }


def should_attach_visual_inputs(query: str, analyzed_intent: Optional[Dict[str, Any]] = None) -> bool:
    intent = analyzed_intent or analyze_query_intent(query)
    return bool(intent.get("wants_visual"))


def should_attach_pdf_binary(query: str, analyzed_intent: Optional[Dict[str, Any]] = None) -> bool:
    intent = analyzed_intent or analyze_query_intent(query)
    if bool(intent.get("wants_full_document")):
        return True
    return intent.get("intent") in {"summary", "compliance"}


def score_chunks_for_query(
    chunks: Sequence[Dict[str, Any]],
    query: str,
    embedding_client: Optional[Any] = None,
    embedding_model: str = "",
    embedding_enabled: bool = False,
    document_id: str = "",
    persist_directory: str = "",
    top_k: int = 32,
    max_chars: int = 24000,
    candidate_pool: int = 48,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected, meta = retrieve_hybrid_chunks(
        document_id=document_id or "pdf_doc",
        chunks=chunks,
        query=query,
        embedding_client=embedding_client if embedding_enabled else None,
        embedding_model=embedding_model if embedding_enabled else "",
        persist_directory=persist_directory,
        top_k=max(1, int(top_k)),
        max_chars=max_chars,
        candidate_pool=max(1, int(candidate_pool)),
    )
    return selected, meta


def merge_page_and_semantic_candidates(
    *,
    query: str,
    scored_chunks: Sequence[Dict[str, Any]],
    page_focus_chunks: Sequence[Dict[str, Any]],
    page_hints: Sequence[int],
    top_k: int,
    max_chars: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Compatibility wrapper: in the new hybrid retriever, scored_chunks are already fused.
    del query, page_focus_chunks, page_hints

    selected: List[Dict[str, Any]] = []
    consumed = 0
    cap = max(500, int(max_chars))

    for chunk in scored_chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        if selected and consumed + len(text) > cap:
            break
        selected.append(dict(chunk))
        consumed += len(text)
        if len(selected) >= max(1, int(top_k)):
            break

    top_score = float(selected[0].get("_score") or 0.0) if selected else 0.0
    second_score = float(selected[1].get("_score") or 0.0) if len(selected) > 1 else 0.0
    gap = max(0.0, top_score - second_score)

    return selected, {
        "strategy": "hybrid_vector_bm25",
        "selected": len(selected),
        "total_candidates": len(scored_chunks),
        "page_focus_candidates": 0,
        "semantic_candidates": len(scored_chunks),
        "confidence": _clamp((0.68 * top_score) + (0.32 * gap), 0.0, 1.0),
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
    document_id: str = "",
    persist_directory: str = "",
    candidate_pool: int = 24,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return retrieve_hybrid_chunks(
        document_id=document_id or "pdf_doc",
        chunks=chunks,
        query=query,
        embedding_client=embedding_client if embedding_enabled else None,
        embedding_model=embedding_model if embedding_enabled else "",
        persist_directory=persist_directory,
        top_k=top_k,
        max_chars=max_chars,
        candidate_pool=candidate_pool,
    )


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
