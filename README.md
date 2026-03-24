# The AI Council

**The AI Council** is a sophisticated multi-LLM chat interface that allows users to consult with a diverse panel of AI models simultaneously. It orchestrates conversations across various providers (Anthropic, DeepSeek, Google, xAI, etc.) through a unified, real-time web interface.

## 🌟 Key Features

*   **Unified Council Interface**: Interact with multiple AI models in a single session. Compare responses or have them work together on complex tasks.
*   **Broad Model Support**: Configurable support for top-tier models including:
    *   **Anthropic**: Claude 3.5/3.7 (Sonnet, Haiku, Opus) and "Thinking" variants.
    *   **DeepSeek**: V3, R1.
    *   **xAI**: Grok 3 (including DeepSearch).
    *   **Google**: Gemini 1.5/2.0 (Pro, Flash).
    *   **OpenAI**: GPT-4o, o1, o3-mini.
*   **Document & Image Analysis**: 
    *   **PDF Support**: Upload PDF documents and pass them directly to PDF-capable models.
    *   **Model Fallback Notice**: If a selected model is not PDF-capable, the system adds a prompt notice that PDFs were attached but cannot be read by that model.
    *   **OCR Capability**: Extract text from uploaded images (JPG, PNG) using `easyocr`.
*   **Web Search Integration**: Dedicated autonomous web search capabilities (defaulting to Grok 3 DeepSearch).
*   **Flexible Configuration**: Easily manage available bots, pricing tiers, and capabilities via `config.json`.
*   **Real-time Streaming**: Low-latency token streaming using Socket.IO.
*   **Chat Management**: Auto-saving chat history and daily temporary sessions.

## 🛠️ Technology Stack

*   **Backend**: Python (Flask, Flask-SocketIO)
*   **Frontend**: HTML5, CSS3, JavaScript (Socket.IO client, Markdown rendering, KaTeX for math)
*   **AI Integration**: OpenAI-compatible client (supporting various endpoints)
*   **Utilities**: `easyocr`, `Pillow`

## 🚀 Getting Started

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configuration**:
    *   Copy `.env.example` to `.env` and add your API keys.
    *   Copy `config.json.example` to `config.json` to customize your available bots.
4.  **Run the Application**:
    *   Execute `main.bat` (Windows)
    *   Or run directly: `python main.py`
5.  **Access the Interface**:
    *   Open your browser to `http://localhost:5000`

## 📝 Usage

*   **Select Bots**: Choose which members of the "Council" you want to engage for a specific query.
*   **Upload Context**: Drag and drop files to provide context to the active models.
*   **Chat**: Send your prompt and watch as multiple perspectives stream in simultaneously.

## 📚 Leader Skills Folder

You can add local markdown skill files that help the Leader decide task routing.

*   Create a folder named `skills/` in the project root.
*   Add one or more `.md` files with routing rules, decision patterns, or role assignment hints.
*   A dedicated **MarkReader** role runs before Leader and selects one or more relevant markdown files.
*   The selected files are loaded into **Leader prompts only** (task distribution and final synthesis), not other roles.

Configure behavior in `config.json` with role models only by default. MarkReader and skills tuning are hardcoded in backend defaults.

```json
"MarkReader": "gpt-4o",
"Leader": "gpt-4o",
"Researcher": "gpt-4o",
"Creator": "claude-3-5-sonnet",
"Analyzer": "gpt-4o",
"Verifier": "gpt-4o"
```

Optional advanced overrides are still supported if you want to tune behavior:

```json
"md_reader": {
    "enabled": true,
    "max_inventory_files": 40,
    "preview_lines_per_file": 20,
    "preview_chars_per_file": 1200
},
"skills": {
    "enabled": true,
    "folder": "skills",
    "max_files": 3,
    "max_chars_per_file": 2500,
    "max_total_chars": 7000
}
```

Runtime notes:
*   If MarkReader is disabled, misconfigured, or inventory is unavailable, council execution continues normally.
*   Console logs report MarkReader selected files, rejected files, and loaded Leader context files.
