# PDF Retrieval Benchmark Scaffold

This folder provides a lightweight benchmark harness for multimodal-hybrid retrieval quality and token efficiency checks.

## Files

- `dataset.template.json`: Example benchmark dataset schema.
- `run_benchmark.py`: Runs the multimodal parser + hybrid retriever on benchmark cases and prints summary metrics.

## Dataset Format

Top-level JSON object:

```json
{
  "cases": [
    {
      "id": "policy-manual-v1",
      "pdf_path": "uploads/manual.pdf",
      "questions": [
        {
          "query": "What does page 70 say about retention?",
          "expected_pages": [70]
        }
      ]
    }
  ]
}
```

## Run

```bash
python eval/pdf_retrieval_benchmark/run_benchmark.py --dataset eval/pdf_retrieval_benchmark/dataset.template.json
```

Optional flags:

- `--top-k`: Selected chunk count limit (default: `8`)
- `--max-chars`: Retrieval excerpt budget (default: `12000`)
- `--candidate-multiplier`: Candidate pool multiplier before final selection (default: `3`)

## Metrics

- `cases_total`: Total question count.
- `hit_at_k`: Questions where at least one expected page appears in selected chunks.
- `recall_at_k`: `hit_at_k / cases_total`.
- `avg_selected_chars`: Average characters kept in selected chunks.
