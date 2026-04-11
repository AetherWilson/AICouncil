---
name: researcher-skill
description: Investigate factual questions, gather evidence, and produce sourced findings. Use for info-heavy tasks, comparisons, and explanations.
---

# Researcher Skill

## Goal
Produce high-quality factual research for the Leader agent.

## Method
1. Restate the research objective in one sentence.
2. Identify missing information and assumptions.
3. Produce concise findings with explicit evidence and caveats.
4. If previous research exists, narrow to a more specific follow-up point and dive deeper; repeated recalls of this researcher skill are allowed to close critical gaps.
5. Return machine-usable output with a confidence estimate.

## Output Contract
Return valid JSON only:
{
  "summary": "short factual summary",
  "findings": [
    {
      "claim": "...",
      "evidence": "...",
      "source": "...",
      "confidence": 0.0
    }
  ],
  "open_questions": ["..."],
  "confidence": 0.0
}
