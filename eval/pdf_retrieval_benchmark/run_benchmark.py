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


def _hit_expected_pages(selected_chunks: Sequence[Dict], expected_pages: Sequence[int]) -> bool:
    expected = {int(p) for p in expected_pages if int(p) > 0}
    if not expected:
        return False

    for chunk in selected_chunks:
        page_number = chunk.get("page_number")
        if isinstance(page_number, int) and page_number in expected:
            return True

        start = int(chunk.get("page_start") or 0)
        end = int(chunk.get("page_end") or start)
        if start > 0 and any(start <= page <= end for page in expected):
            return True

    return False


def _default_doc_settings() -> Dict[str, object]:
    return {
        "pdf_layout_model": "",
        "pdf_layout_dpi": 180,
        "pdf_layout_max_dimension": 2200,
        "pdf_layout_max_total_bytes": 15728640,
        "pdf_layout_max_pages": 200,
        "pdf_image_description_enabled": False,
        "pdf_image_description_model": "",
        "pdf_image_description_max_images_per_page": 0,
        "pdf_image_description_max_total_images": 0,
        "pdf_image_min_bytes": 4096,
        "pdf_chunk_max_chars": 1800,
        "pdf_chunk_max_items": 400,
        "pdf_text_max_chars": 500000,
    }


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

        case_id = str(case.get("id") or "case")
        pdf_path = str(case.get("pdf_path") or "").strip()
        if not pdf_path:
            continue

        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(ROOT_DIR, pdf_path)

        if not os.path.isfile(pdf_path):
            print(f"[WARN] PDF not found: {pdf_path}")
            continue

        parsed = pdf_parser.parse_pdf_document(pdf_path, _default_doc_settings())
        chunks = parsed.get("chunks") if isinstance(parsed.get("chunks"), list) else []
        questions = case.get("questions") if isinstance(case.get("questions"), list) else []

        for question in questions:
            if not isinstance(question, dict):
                continue

            query = str(question.get("query") or "").strip()
            expected_pages = question.get("expected_pages") if isinstance(question.get("expected_pages"), list) else []
            if not query:
                continue

            total_questions += 1
            pool_size = max(top_k, min(120, top_k * max(1, int(candidate_multiplier))))
            selected_chunks, _ = document_retriever.retrieve_hybrid_chunks(
                document_id=f"benchmark:{case_id}",
                chunks=chunks,
                query=query,
                embedding_client=None,
                embedding_model="",
                persist_directory=os.path.join(ROOT_DIR, "uploads", ".pdf_chroma_benchmark"),
                top_k=top_k,
                max_chars=max_chars,
                candidate_pool=pool_size,
                vector_weight=0.0,
                bm25_weight=1.0,
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
    parser = argparse.ArgumentParser(description="Run multimodal PDF retrieval benchmark scaffold")
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
