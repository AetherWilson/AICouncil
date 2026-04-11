---
name: analyzer-skill
description: Perform structured reasoning, calculations, and logic checks for quantitative or constraint-heavy tasks.
---

# Analyzer Skill

## Goal
Provide rigorous analysis and calculations.

## Method
1. Define the problem and variables.
2. Show the minimum required reasoning and computations.
3. State assumptions and sensitivity.
4. Return machine-usable analytical output.

## Output Contract
Return valid JSON only:
{
  "analysis": "core reasoning",
  "calculations": [
    {
      "step": "...",
      "result": "..."
    }
  ],
  "assumptions": ["..."],
  "confidence": 0.0
}
