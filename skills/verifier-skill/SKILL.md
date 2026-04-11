---
name: verifier-skill
description: Verify logic and factual correctness, especially for math and reasoning tasks. Report concrete issues and severity.
---

# Verifier Skill

## Goal
Audit outputs and identify concrete errors, risk, and confidence gaps.

## Method
1. Check factual correctness and internal consistency.
2. Flag only concrete, actionable issues.
3. Suggest corrections where possible.
4. Return machine-usable verification output.

## Output Contract
Return valid JSON only:
{
  "status": "pass_or_warn_or_fail",
  "issues": [
    {
      "severity": "critical|important|minor",
      "issue": "...",
      "fix": "..."
    }
  ],
  "confidence": 0.0
}
