# AI Council

AI Council is a local multi-model chat app that lets you send one prompt to a configurable skill-first agent workflow, stream outputs live, and manage chats through a lightweight web UI.

It is built around a single Leader-agent loop that can select markdown skills, execute optional per-skill scripts, and manage persistent memory, all routed through an OpenAI-compatible API client.

> ## Vibe coded notice
> This project is **vibe coded**: it was built iteratively and fast, with a strong focus on getting useful behavior on screen quickly. Expect practical structure, evolving conventions, and occasional rough edges.

## What it does

- Run a skill-first Leader-agent conversation from one UI
- Stream responses in real time with Socket.IO
- Configure role-to-model assignments in `config.json`
- Manage available models and capabilities in `model.json`
- Upload **PDFs**, **images**, and **Word OpenXML files** as chat context
- Save named chats and auto-save temporary chats
- Maintain persistent cross-chat memory in `skills/memories/memory.md`
- Inspect backend state from the built-in **Backend** tab
- Run a quick **Uptest** that sends a hello request to each configured council role and reports latency/status

## Current architecture

### Backend

- **Python**
- **Flask** for HTTP routes
- **Flask-SocketIO** for live streaming updates
- **OpenAI Python client** with optional custom `base_url`

### Frontend

- Single-page HTML interface in `templates/index.html`
- Markdown rendering via `marked`
- Syntax highlighting via `highlight.js`
- Math rendering via `KaTeX`

### Utilities

- `Pillow` for image handling
- `opencc` for Simplified → Traditional Chinese conversion support
- `python-dotenv` for environment loading
- `httpx` for API transport

## Project structure

Key files and folders:

- `main.py` — main Flask + Socket.IO app
- `GPT_handle.py` — model request handling and streaming helpers
- `services/config_store.py` — cached config/model loading
- `services/memory_manager.py` — persistent memory file management
- `templates/index.html` — main web UI
- `config.json` — active role/model config
- `config.json.example` — starter config
- `model.json` — enabled model catalog and capability metadata
- `skills/` — optional markdown skills and memory files
- `chat_history/` — saved chats
- `temp_chat_history/` — auto-saved temporary chats
- `uploads/` — uploaded PDFs, images, and Word OpenXML files
- `gpt_responses/` — request/response debug logs

## Features in more detail

### 1. Leader-agent workflow

The app now runs a single Leader-agent planning loop. The Leader can:

- call markdown skills discovered from `skills/`
- optionally execute `skills/<skill>/scripts/*.py` tool scripts when planned
- produce a final response
- run memory extraction/writeback through `MemWriter` logic in `main.py`

Key model assignments live in `config.json`.

Example:

```json
{
  "Leader": "gpt-4o",
  "lite_model": "gpt-4o-mini",
  "history_context_mode": "final_only",
  "skills": {
    "enabled": true,
    "folder": "skills",
    "allow_legacy_flat": true,
    "model_map": {
      "researcher-skill": "gpt-4o-mini",
      "creator-skill": "gpt-4o",
      "analyzer-skill": "gpt-4o",
      "verifier-skill": "gpt-4o",
      "pdfer-skill": "gemini-3.1-pro-preview"
    }
  }
}
```

`lite_model` is used for lightweight internal tasks such as markdown skill file selection and memory-model fallback.

`history_context_mode` controls what prior assistant outputs are sent back into later model calls:

- `final_only` (recommended): include only prior Leader final responses plus user turns
- `all`: include all prior assistant outputs (agent steps + intermediate outputs + final)

### 2. Model registry

`model.json` is the source of truth for available models. Each entry can include:

- `id`
- `name`
- `enabled`
- `price`
- `provider`
- `support_images`
- `support_pdf_input`

This makes it easy to expose only the models you actually want available in the UI.

### 3. File uploads

The current app supports:

- `.pdf`
- `.jpg`
- `.jpeg`
- `.png`
- `.docx`
- `.docm`
- `.dotx`
- `.dotm`

Legacy `.doc` files are not supported.

Uploaded files are stored in `uploads/` and registered against the active conversation.

For PDFs:

- upload-time brief is generated automatically by `pdfer-skill` model routing
- Leader reads the brief before deciding whether deeper PDF reads are needed
- for PDF-related asks, Leader is forced to call `pdfer-skill` first
- `pdfer-skill` receives filename-scoped PDF inputs and can be called repeatedly for iterative discovery
- brief metadata (summary/topics/confidence) is stored with each uploaded PDF and persisted with chats

For images:

- images are converted to data URLs for model input where supported

For Word OpenXML files:

- text is extracted on the backend
- extracted content is injected into role context for all models

### 4. Chat persistence

There are two kinds of chat storage:

- **Saved chats** in `chat_history/`
- **Temporary chats** in `temp_chat_history/`

Temporary chats are meant for same-day work and are cleaned up automatically when old.

### 5. Persistent memory

Cross-chat memory is managed through:

- `services/memory_manager.py`
- `skills/memories/memory.md`

The memory system is intended to store useful long-lived context and inject it into later conversations.

### 6. Skills support

Skills are loaded from `skills/` and can provide both markdown instructions and optional local scripts.

The flow is:

1. Persistent memory from `skills/memories/memory.md` is loaded into Leader context by default
2. The Leader-agent can call discovered skills directly from `skills/`
3. A planned local tool call can run `scripts/<name>.py` inside the target skill folder

Local script execution is sandboxed to the skill's own `scripts/` directory and expects JSON on stdout.

### 7. Backend monitoring

The UI includes a **Backend** tab where you can:

- inspect backend-related panels
- run an **Uptest** against configured council models
- view role/model availability and response timing behavior

## Requirements

From `requirements.txt`:

- `flask`
- `flask-socketio`
- `python-socketio`
- `openai`
- `httpx`
- `python-dotenv`
- `Pillow`
- `python-docx`
- `opencc`

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd Council
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```env
gpt_api_key="your-api-key-here"
gpt_redirect_url="your-redirect-url-here"
```

Notes:

- `gpt_api_key` is your API key
- `gpt_redirect_url` can point to an OpenAI-compatible endpoint
- `gpt_redirect_url` can be left empty if you want the default OpenAI endpoint behavior

### 5. Configure council roles

Copy:

- `config.json.example` → `config.json`

Then edit the role-to-model mapping as needed.

### 6. Review enabled models

Edit `model.json` to decide:

- which models are enabled
- provider labels
- price tiers
- image support
- PDF support

## Running the app

### Windows helper

```bat
main.bat
```

### Direct run

```bash
python main.py
```

Then open:

```text
http://localhost:5000
```

## Basic usage

1. Open the web UI
2. Choose the bots/models you want involved
3. Type a prompt
4. Optionally upload PDFs or images
5. Watch responses stream in live
6. Save the chat or let it remain temporary

## Notes and caveats

- This is a **local app-first** workflow, not a polished SaaS product
- The codebase is intentionally flexible and may change quickly
- Uploaded files, saved chats, temp chats, and debug logs are stored locally
- Some behavior depends on whether your configured endpoint supports specific model capabilities
- The project contains practical logging and persistence features, but it is still very much an evolving tool

## License

Add your preferred license here if you plan to distribute the project.
