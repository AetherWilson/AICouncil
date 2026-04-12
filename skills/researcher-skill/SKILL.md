---
name: researcher-skill
description: Investigate factual questions, gather evidence, and produce sourced findings. Default to this skill for info-heavy tasks, comparisons, and explanations whenever there is any uncertainty, even slight uncertainty.
---

# Researcher Skill

## Goal
Produce high-quality factual research for the Leader agent.

Routing intent for the Leader agent:
- If there is even a small chance the answer may be incomplete, stale, assumption-heavy, or weakly supported, call this skill.
- Treat "probably correct" as insufficient for skipping this skill on factual requests.

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
