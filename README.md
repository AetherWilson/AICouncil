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
    *   **PDF Support**: Upload and analyze PDF documents.
    *   **OCR Capability**: Extract text from uploaded images (JPG, PNG) using `easyocr`.
*   **Web Search Integration**: Dedicated autonomous web search capabilities (defaulting to Grok 3 DeepSearch).
*   **Flexible Configuration**: Easily manage available bots, pricing tiers, and capabilities via `config.json`.
*   **Real-time Streaming**: Low-latency token streaming using Socket.IO.
*   **Chat Management**: Auto-saving chat history and daily temporary sessions.

## 🛠️ Technology Stack

*   **Backend**: Python (Flask, Flask-SocketIO)
*   **Frontend**: HTML5, CSS3, JavaScript (Socket.IO client, Markdown rendering, KaTeX for math)
*   **AI Integration**: OpenAI-compatible client (supporting various endpoints)
*   **Utilities**: `PyPDF2`, `easyocr`, `Pillow`

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
