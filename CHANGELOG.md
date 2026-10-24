# Changelog

## 0.1.0

Version numbering restarts at 0.1.0. The major component changes only when the
user explicitly designates a major release.

- Rebuild the UI around a responsive ChatGPT-style conversation layout, with
  automatic permanent chat saving, history search/rename/delete, themes,
  file attachments, edit/regenerate, stop and expandable council records.
- Add Codex-inspired context checkpoints: automatic budgeting, recent user input,
  protected current requests/drafts, bounded summarization, durable checkpoints,
  edit invalidation and one context-overflow recovery attempt.
- Continue output-limited long replies on the same scheduled model without
  releasing its turn; preserve the full answer while compacting working input.
- Keep private ballot isolation, ordered member turns, revision limits and
  majority thresholds intact.
- Preserve immutable attachment versions so history edits restore their original
  sources and same-named uploads in different chats remain isolated.
- Add deterministic tests for compaction, continuation, persistence, cancellation,
  exact draft preservation and attachment isolation; validate desktop and mobile
  flows in Chromium using simulated models.
