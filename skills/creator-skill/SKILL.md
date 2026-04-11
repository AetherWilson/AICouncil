---
name: creator-skill
description: Create polished drafts, structured writing, and presentation-ready outputs from requirements.
---

# Creator Skill

## Goal
Transform requirements and research into clear, high-quality output.

## Method
1. Extract audience, tone, constraints, and deliverable format.
2. Produce output that is directly usable by the user.
3. Keep style consistent and avoid filler.
4. Return machine-usable output with confidence.

## Output Contract
Return valid JSON only:
{
  "content": "final drafted content",
  "format": "markdown",
  "notes": ["..."],
  "confidence": 0.0
}
