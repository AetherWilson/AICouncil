# AI Council

AI Council is a local multi-model chat app built with Python, Flask and Socket.IO.
Every member works on the **whole problem**, without a specialist identity. Members
submit independent proposals, discuss in a fixed speaking order, and review the
chair's draft before it becomes the final answer.

## Council procedure

1. **Independent proposals.** Every enabled seat, including the chair, submits a
   complete proposal. Each receives the same prior conversation, source files and
   initial instructions, without seeing the other proposals.
2. **Ordered discussion.** Seats speak one at a time in the order of `council.members`.
   Each seat gets exactly one turn per round and can address any aspect of the
   problem. Later speakers see completed turns. A faster model never gets extra turns.
   Discussion ends after a full round if nobody requests another round, or at the
   configured round limit.
3. **Chair draft.** The chair combines the useful ideas, evidence, uncertainties and
   unresolved disagreements into a complete answer.
4. **Private review.** Non-chair members review the exact draft, in seat order.
   Each submits `approve`, `request_changes`, `reject` or `abstain`, with a reason.
   Current ballots are hidden from subsequent voters and the UI until all reviews
   finish. The chair does not vote on its own draft.
5. **Revision and re-vote.** Approval requires **more than half of all enabled
   non-chair seats**. Failed, timed-out, malformed or abstaining ballots do not
   reduce that denominator. A rejected draft is revised and the entire electorate
   votes again, up to `max_revisions`.
6. **Final answer.** The approved draft is preserved verbatim, with the result and
   ballot reasons shown alongside it. There is no additional model rewrite after
   approval. If no draft passes, the final output is explicitly marked **Not approved**.
   Majority agreement is not treated as factual proof.

Every turn has a total time limit, including compaction and continuation requests. A failed seat is marked and skipped; late output
cannot enter a later turn. Stop cancels the sitting and prevents further turns or
an approved final answer. The controller may finish before a blocked provider
connection closes, but output from that connection is discarded.

During generation, type a correction and press **Enter** or choose **送出補充**. Updates are
applied at the next discussion/draft boundary so the independent proposal inputs
stay consistent. An update received during drafting or voting invalidates approval
of that draft and requires a revision and new ballot. If the revision budget is
exhausted, the result remains Not approved.

## Chat interface (0.1.0)

The interface follows the familiar ChatGPT layout: conversation history on the
left, a centered transcript, and a rounded composer at the bottom. New conversations
are saved automatically as permanent local chats. Search, rename and delete chats
from the sidebar; switch light/dark themes from the header. Existing temporary
chats remain accessible, and renaming one saves it permanently.

- Enter sends; Shift+Enter inserts a newline. IME composition does not send early.
- Attach PDF, image or Word files with **+**, drag/drop, or paste an image.
- Use the stop button to cancel a response; during work, Enter submits a user update.
- Edit a user message or regenerate a final answer to start another council sitting.
  Editing an earlier message discards subsequent turns and restores that message's
  original attachment snapshot.
- The council discussion and ballot reasons are expandable. The accepted answer
  remains the primary content. Markdown, code copying, tables and math are supported;
  rendered model HTML is sanitized.
- **AI Council** in the header opens member order, manual memory management, and
  connection diagnostics. Mobile screens use a slide-out history panel.
- The context indicator shows recent input use and lets you request compaction
  before the next model call. Disconnect/reconnect restores the active sitting.

## Context checkpoints and long responses

The context manager follows the local-summary pattern in the public
[Codex CLI compaction implementation](https://github.com/openai/codex/blob/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/core/src/compact.rs):
compress earlier working context, retain recent user input, re-inject the current
instructions, save a checkpoint and continue. This app implements that pattern
through its OpenAI-compatible Chat Completions transport.

Full chat messages are never replaced by summaries on disk or in the UI. A separate
checkpoint stores the summary, covered prefix and a content hash. Reloading a chat
reuses a valid checkpoint; editing covered history invalidates it. Summaries preserve
goals, constraints, decisions, evidence references, unresolved disagreements and next
steps. They do not become persistent user memories.

The current user request, updates, original attachments and the **exact current
draft under review** are protected. Earlier council contributions can be compacted,
but current private ballots never enter another voter's context. Every request
receives the canonical phase instructions again. A common budget based on the
smallest configured member window avoids giving a faster model extra context or
turns. Compaction itself uses the scheduled seat's model, with no helper model or
specialist identity. Large summarization inputs are processed in bounded fragments.
Checkpoints are committed only after successful summarization; errors/cancellation
keep the prior checkpoint and original transcript.

If the provider reports context overflow, the same seat gets at most one forced
compaction/retry. If the current request or protected sources still cannot fit, the
turn fails explicitly rather than silently omitting them. Provider output ending
with `finish_reason=length` is continued by the same seat with the already-emitted
prefix. Very long emitted prefixes can themselves be handed off as a summary plus
an exact recent tail; the full emitted answer stays intact. The seat keeps its place
in the speaking order through these requests. Incomplete answers are not accepted
as complete drafts or ballots after the configured continuation/time limit.

Add or adjust `context` in `config.json` (defaults also apply to older configs):

```json
{
  "context": {
    "enabled": true,
    "context_window_tokens": 128000,
    "auto_compact_ratio": 0.8,
    "reserve_output_tokens": 8192,
    "summary_tokens": 2048,
    "recent_history_tokens": 6000,
    "recent_council_records": 3,
    "pdf_token_reserve": 16000,
    "image_token_reserve": 2048,
    "max_continuations": 4
  }
}
```

Model entries in `model.json` can override `context_window_tokens` and
`max_output_tokens`. Supply the actual limits for your endpoint. Token counts use
`tiktoken` where its encoding data is available, and a conservative UTF-8 estimate
otherwise. Unknown model tokenizers and original PDF/image costs are estimates;
`pdf_token_reserve` and `image_token_reserve` are configurable per-file reservations.
Actual usage reported by the provider takes precedence in the indicator. A reserve
for answer output is always left available. Compaction and continuation requests
consume the same seat's turn timeout and can incur ordinary API usage.

New uploads receive immutable storage IDs while retaining their display filenames,
including Unicode names. Same-named files in different chats cannot overwrite one
another. Removing an attachment detaches it from future input; the source remains
available to earlier messages for regeneration. Legacy filename-based attachments
can still be restored.

## Setup

```bash
git clone https://github.com/AetherWilson/AICouncil.git
cd AICouncil
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your OpenAI-compatible endpoint:

```env
gpt_api_key="your-api-key"
gpt_redirect_url=""
```

Copy `config.json.example` to `config.json`, then select your models:

```json
{
  "history_context_mode": "final_only",
  "council": {
    "chair": "chair",
    "members": [
      {"id": "chair", "name": "Chair", "model": "gpt-4o", "enabled": true},
      {"id": "member-1", "name": "Member 1", "model": "gpt-4o-mini", "enabled": true},
      {"id": "member-2", "name": "Member 2", "model": "gpt-4o", "enabled": true}
    ],
    "discussion_rounds": 2,
    "max_revisions": 1,
    "turn_timeout_seconds": 120
  },
  "memory": {
    "enabled": true,
    "auto_extract": false,
    "path": "skills/memories/memory.md"
  }
}
```

- The member list determines speaking order. The chair can appear anywhere in it.
- At least two enabled seats are required: the chair and one reviewer. IDs must
  be unique, begin with a letter and contain only letters, digits, `_` or `-` (up
  to 64 characters). Models may be reused for different seats, although selecting
  different models can provide more varied proposals.
- `discussion_rounds` is the maximum number of discussion rounds after proposals
  (1–8). `max_revisions` counts revisions after the initial draft (0–5).
- `turn_timeout_seconds` limits the entire turn, including streaming (1–600).
  There are no helper-model fallbacks. A provider context-overflow error can trigger one compaction/retry on the same model; output-limit continuations are bounded separately.
- `history_context_mode` can be `final_only` (prior user turns and final answers)
  or `all` (also include previous council contributions). User updates are retained
  in either mode. Each sitting uses a snapshot of history and immutable source files.
- `model.json` remains an optional catalog for model display names and the Models
  API. The council uses the exact model IDs in its seat configuration.

Run `python main.py` (or `main.bat` on Windows), then open
<http://localhost:5000>. The Backend tab shows the roster in speaking order and
includes a user-triggered Uptest for checking connectivity to each seat's model.

## Files and memory

PDFs and images are attached **directly to every model request**, including draft
reviews. The configured models/endpoint are assumed to support native PDF and
image inputs through the OpenAI-compatible API. Capability flags do not route
files through another model. A missing/unreadable file fails the turn instead of
silently answering without that source.

Word OpenXML files (`.docx`, `.docm`, `.dotx`, `.dotm`) are extracted locally with
`python-docx` and included as text. Legacy `.doc` files are unsupported. Uploads
are limited to 16 MB per request.

The former researcher, creator, analyzer, verifier and PDF-reader skills, skill
selection models, upload-time PDF summaries and model-based memory extraction
have been removed. Context compaction uses the scheduled member model to summarize
old text; it does not introduce another agent or a file-reading model. Uploading/restoring a PDF makes **no LLM request**. Existing
persistent memory can still be read into council context and managed manually
through the Memory UI/API or the configured file. `auto_extract` is always false.

## Upgrading an older installation

An existing `config.json` without a `council` object is adapted in memory:
`Leader` becomes the chair; existing specialist model assignments become neutral
numbered seats in their former order. Duplicate specialist model IDs are grouped
into one seat. If no reviewer model is configured, one neutral reviewer uses the
chair model. The old PDF reader is not promoted to a seat. Existing settings and
credentials are not overwritten; use the new config format to explicitly choose
your electorate and speaking order. Old `skills`, `lite_model` and `agent_loop`
settings no longer execute anything.

Old saved chats remain readable. New records retain member/model identity,
phase, round, draft version, ballots and approval status. The UI groups the council
record in an expandable **議會討論** area and highlights the final answer.

## Project structure

- `main.py`: HTTP/Socket.IO routes, ordered-turn streaming, attachments and persistence
- `services/council_workflow.py`: proposals, discussion, drafting and private voting
- `services/config_store.py`: seat validation, defaults and legacy configuration adapter
- `services/context_manager.py`: token budgets, checkpoints and protected request context
- `services/memory_manager.py`: local persistent memory and manual editing
- `GPT_handle.py`: OpenAI-compatible requests and native file inputs
- `templates/index.html`, `static/chat.js`, `static/chat.css`: responsive chat UI and council record
- `tests/`: deterministic workflow and Flask integration tests with simulated models

Chats (`chat_history/`, `temp_chat_history/`), uploads, memory, credentials and
request logs (`gpt_responses/`) stay local and are excluded from Git. Old temporary
chats are cleaned up automatically on subsequent days.

## Validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py GPT_handle.py services tests
```

Tests use simulated models and do not require a paid API request. They cover
ordered turns, independent proposal inputs, hidden ballots, majority thresholds,
revisions, failure/timeout/cancellation handling, direct PDF inputs, metadata
persistence and old chat compatibility. Real model quality and endpoint file
support depend on the selected models and provider.
