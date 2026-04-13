import argparse
import json
import os
import sys
from typing import Dict, List, Sequence


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services import document_retriever, pdf_parser  # noqa: E402


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Dataset must be a JSON object")
    return payload


def _build_page_focus_chunks(native_pages: Sequence[Dict], target_pages: Sequence[int], max_chars: int) -> List[Dict]:
    page_map = {}
    for item in native_pages:
        page_num = int(item.get("page") or 0)
        text = str(item.get("text") or "").strip()
        if page_num > 0 and text:
            page_map[page_num] = text

    chunks: List[Dict] = []
    consumed = 0
    for page in sorted(set(int(p) for p in target_pages if int(p) > 0)):
        text = page_map.get(page, "")
        if not text:
            continue
        remaining = max_chars - consumed
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        chunks.append(
            {
                "text": text,
                "page_start": page,
                "page_end": page,
                "sources": ["archive_native_text"],
                "char_count": len(text),
                "block_count": 1,
                "_score": 0.32,
            }
        )
        consumed += len(text)

    return chunks


def _hit_expected_pages(selected_chunks: Sequence[Dict], expected_pages: Sequence[int]) -> bool:
    expected = {int(p) for p in expected_pages if int(p) > 0}
    if not expected:
        return False

    for chunk in selected_chunks:
        start = int(chunk.get("page_start") or 0)
        end = int(chunk.get("page_end") or start)
        if start <= 0:
            continue
        if any(start <= page <= end for page in expected):
            return True
    return False


def run_benchmark(dataset: Dict, top_k: int, max_chars: int, candidate_multiplier: int) -> Dict:
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Dataset field 'cases' must be a list")

    total_questions = 0
    hits = 0
    selected_char_sum = 0

    for case in cases:
        if not isinstance(case, dict):
            continue
        pdf_path = str(case.get("pdf_path") or "").strip()
        if not pdf_path:
            continue

        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(ROOT_DIR, pdf_path)

        if not os.path.isfile(pdf_path):
            print(f"[WARN] PDF not found: {pdf_path}")
            continue

        parsed = pdf_parser.parse_pdf_document(pdf_path)
        chunks = parsed.get("chunks") if isinstance(parsed.get("chunks"), list) else []
        native_pages = parsed.get("native_pages") if isinstance(parsed.get("native_pages"), list) else []
        questions = case.get("questions") if isinstance(case.get("questions"), list) else []

        for question in questions:
            if not isinstance(question, dict):
                continue
            query = str(question.get("query") or "").strip()
            expected_pages = question.get("expected_pages") if isinstance(question.get("expected_pages"), list) else []
            if not query:
                continue

            total_questions += 1
            page_hints = document_retriever.extract_page_hints(query, max_pages=8)
            scored_chunks, _ = document_retriever.score_chunks_for_query(
                chunks,
                query=query,
                embedding_client=None,
                embedding_model="",
                embedding_enabled=False,
            )

            pool_size = max(top_k, min(50, top_k * max(1, int(candidate_multiplier))))
            semantic_candidates = scored_chunks[:pool_size]
            page_focus_chunks = _build_page_focus_chunks(native_pages, page_hints, max_chars=max_chars // 2)

            selected_chunks, _ = document_retriever.merge_page_and_semantic_candidates(
                query=query,
                scored_chunks=semantic_candidates,
                page_focus_chunks=page_focus_chunks,
                page_hints=page_hints,
                top_k=top_k,
                max_chars=max_chars,
            )

            selected_char_sum += sum(len(str(item.get("text") or "")) for item in selected_chunks)
            if _hit_expected_pages(selected_chunks, expected_pages):
                hits += 1

    recall = (hits / total_questions) if total_questions else 0.0
    avg_selected_chars = (selected_char_sum / total_questions) if total_questions else 0.0

    return {
        "cases_total": total_questions,
        "hit_at_k": hits,
        "recall_at_k": round(recall, 4),
        "avg_selected_chars": round(avg_selected_chars, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDF retrieval benchmark scaffold")
    parser.add_argument("--dataset", required=True, help="Path to benchmark dataset JSON")
    parser.add_argument("--top-k", type=int, default=8, help="Final chunk selection count")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max selected excerpt chars")
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=3,
        help="Candidate pool multiplier before final selection",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(ROOT_DIR, dataset_path)

    dataset = _load_json(dataset_path)
    result = run_benchmark(
        dataset,
        top_k=max(1, args.top_k),
        max_chars=max(500, args.max_chars),
        candidate_multiplier=max(1, args.candidate_multiplier),
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
