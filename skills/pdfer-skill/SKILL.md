---
name: pdfer-skill
description: PDF reader. Read only the requested uploaded PDF filenames, extract relevant facts, and return concise evidence-backed findings.
---

# PDF Reader Skill

## Goal
Answer questions using only the selected uploaded PDFs.

## Input Expectations
- `args.task`: the user question to answer
- `args.filenames`: list of uploaded PDF filenames to read

## Rules
1. Read only PDFs in `args.filenames`.
2. If filename scope is missing or empty, return an error result in JSON.
3. Be concise and evidence-focused.
4. Prefer direct citations by filename.

## Output Contract
Return valid JSON only:
{
  "summary": "short answer",
  "findings": [
    {
      "filename": "source.pdf",
      "claim": "...",
      "evidence": "..."
    }
  ],
  "open_questions": ["..."],
  "confidence": 0.0
}
