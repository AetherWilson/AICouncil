from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room
import json
import time
import threading
import uuid
from datetime import datetime
from GPT_handle import completion_response_stream, convert_to_traditional_chinese
import os
from werkzeug.utils import secure_filename
import PyPDF2
from PIL import Image
import base64
from services.config_store import ConfigStore, get_model_info as resolve_model_info, infer_model_support_images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHAT_HISTORY_FOLDER'] = 'chat_history'
app.config['TEMP_CHAT_HISTORY_FOLDER'] = 'temp_chat_history'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")

COUNCIL_ROLES = ['Leader', 'Researcher', 'Creative_Writer', 'Analyzer', 'Verifier']
config_store = ConfigStore(base_dir='.')

# Store all conversation state keyed by conversation_id
# {
#   conversation_id: {
#     messages: list,
#     is_generating: bool,
#     active_stream_task: handle,
#     pending_message_id: str | None,
#     abort_event: threading.Event,
#     uploaded_documents: dict[str, dict]
#   }
# }
conversations = {}

# Prevent concurrent writes from overlapping background tasks.
persistence_lock = threading.Lock()


def _new_conversation_state():
    return {
        'messages': [],
        'is_generating': False,
        'active_stream_task': None,
        'pending_message_id': None,
        'abort_event': threading.Event(),
        'uploaded_documents': {},
        'run_group_counter': 0,
        'current_run_group_id': None
    }


def get_conversation(conversation_id):
    if not conversation_id:
        return None
    if conversation_id not in conversations:
        conversations[conversation_id] = _new_conversation_state()
    return conversations[conversation_id]


def append_conversation_message(conversation_id, role, content, **extra):
    conv = get_conversation(conversation_id)
    message = {
        'id': extra.get('id', uuid.uuid4().hex),
        'role': role,
        'content': content,
        'timestamp': extra.get('timestamp', datetime.now().isoformat())
    }
    for key, value in extra.items():
        if key not in ('id', 'timestamp'):
            message[key] = value
    conv['messages'].append(message)
    return message


def update_message_content(conversation_id, message_id, content):
    conv = get_conversation(conversation_id)
    for msg in conv['messages']:
        if msg.get('id') == message_id:
            msg['content'] = content
            return True
    return False


def update_message_fields(conversation_id, message_id, **fields):
    conv = get_conversation(conversation_id)
    for msg in conv['messages']:
        if msg.get('id') == message_id:
            msg.update(fields)
            return True
    return False


def get_message_content(conversation_id, message_id):
    conv = get_conversation(conversation_id)
    for msg in conv['messages']:
        if msg.get('id') == message_id:
            return msg.get('content', '')
    return ''


def should_continue_streaming(conversation_id, pending_message_id):
    conv = get_conversation(conversation_id)
    return (
        conv is not None
        and conv.get('is_generating', False)
        and not conv.get('abort_event').is_set()
        and conv.get('pending_message_id') == pending_message_id
    )

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Create chat history directory if it doesn't exist
os.makedirs(app.config['CHAT_HISTORY_FOLDER'], exist_ok=True)
# Create temp chat history directory if it doesn't exist
os.makedirs(app.config['TEMP_CHAT_HISTORY_FOLDER'], exist_ok=True)

# Load council role configurations
def load_config():
    return config_store.load_config()

def load_models():
    """Load available models from model.json"""
    return config_store.load_models()

def get_model_info(model_id):
    """Get model details by ID from model.json"""
    return resolve_model_info(load_models(), model_id)

# Council role system prompts
LEADER_DISTRIBUTE_PROMPT = """You are the Leader of an AI council. Your job is to analyze the user's request and distribute tasks to your team members.

Your team:
- Researcher: Gathers factual information, grounds responses with data, finds relevant facts and sources
- Creative_Writer: Writes creative content, generates engaging narratives, provides creative ideas and solutions
- Analyzer: Performs calculations, data analysis, and mathematical reasoning, does not have web access and should not research facts
- Verifier: Reviews and fact-checks all other team members' outputs for accuracy, logical errors, and consistency

Analyze the user's request and create specific tasks for each team member. If a team member is not needed for this request, set their task to "SKIP". The Verifier should almost always be assigned unless nothing needs checking.

You MUST respond with ONLY a valid JSON object in this exact format (no markdown code blocks, no explanation, just raw JSON):
{
    "analysis": "Your brief analysis of what the user needs",
    "researcher_task": "Specific task for the researcher, or SKIP",
    "creative_writer_task": "Specific task for the creative writer, or SKIP",
    "analyzer_task": "Specific task for the analyzer, or SKIP",
    "verifier_task": "Specific task for the verifier (what to check), or SKIP"
}"""

RESEARCHER_PROMPT = """You are the Researcher in an AI council. Gather only the facts directly relevant to the task — nothing more.

- Be concise: key facts, figures, and sources only
- Cite URLs where available
- No background padding or preamble

Do NOT prefix your response with your role name."""

CREATIVE_WRITER_PROMPT = """You are the Creative Writer in an AI council. Produce concise, high-quality creative content for the task.

- Include only essential content — no filler or fluff
- Match tone and style to the context
- Use markdown formatting

Do NOT prefix your response with your role name."""

ANALYZER_PROMPT = """You are the Analyzer in an AI council. Perform calculations and quantitative analysis only — do NOT review other members' work.

- Show working only for key steps, not every trivial detail
- State results clearly and precisely
- Be concise; omit narrative unless it clarifies the method

Do NOT prefix your response with your role name."""

VERIFIER_PROMPT = """You are the Verifier in an AI council. Your ONLY job is to report errors — do NOT comment on what is correct.

Rules:
- List ONLY problems: inaccuracies, logical errors, unsupported claims, contradictions, or flawed calculations
- Do NOT say anything is good, accurate, or fine — silence means you found no issue with that part
- One bullet per issue; be specific about what is wrong and why
- If a claim requires data beyond your knowledge cutoff, state only that specific point is unverifiable — do not speculate
- If you find zero issues overall, output a single line: "No issues found."

Do NOT prefix your response with your role name."""

LEADER_COMBINE_PROMPT = """You are the Leader of an AI council. Your team has completed their work and the Verifier has flagged any errors. Where a member was challenged, they have provided an updated response — always treat that updated response as their final contribution.

Synthesize into a single cohesive final response:
- Use the most current version of each contribution (updated responses take priority over originals)
- Incorporate any Verifier corrections
- Be well-organized and directly address the user's request
- Use proper markdown formatting

Respond directly — do NOT mention the council, team members, or that you are combining results. Do NOT prefix with your role name."""

def cleanup_old_temp_chats():
    """Remove temporary chats from previous days"""
    try:
        temp_folder = app.config['TEMP_CHAT_HISTORY_FOLDER']
        current_date = datetime.now().date()
        
        for filename in os.listdir(temp_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(temp_folder, filename)
                # Get file modification time
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
                
                # Remove if file is from a previous day
                if file_mtime < current_date:
                    os.remove(filepath)
                    print(f"Removed old temporary chat: {filename}")
    except Exception as e:
        print(f"Error cleaning up temporary chats: {e}")


def _extract_document_payload(filepath, filename):
    """Extract normalized metadata/content for a stored document file."""
    file_size = os.path.getsize(filepath)
    file_ext = filename.lower().split('.')[-1]

    if file_ext == 'pdf':
        text_content = extract_pdf_text(filepath)
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            page_count = len(pdf_reader.pages)
        return {
            'filename': filename,
            'content': text_content,
            'pages': page_count,
            'size': file_size,
            'type': 'pdf'
        }

    img = Image.open(filepath)
    width, height = img.size
    return {
        'filename': filename,
        'content': '[Image uploaded]',
        'filepath': filepath,
        'width': width,
        'height': height,
        'size': file_size,
        'type': 'image'
    }


def _build_document_response(doc_payload):
    """Return the UI-safe document payload without internal-only fields."""
    return {
        'filename': doc_payload.get('filename'),
        'type': doc_payload.get('type'),
        'size': doc_payload.get('size'),
        'pages': doc_payload.get('pages'),
        'width': doc_payload.get('width'),
        'height': doc_payload.get('height')
    }

@app.route('/')
def index():
    # Clean up old temporary chats on page load
    cleanup_old_temp_chats()
    return render_template('index.html')

@app.route('/api/council', methods=['GET'])
def get_council():
    """Get council role configuration with model details"""
    config = load_config()
    roles = {}
    for role_name in ['Leader', 'Researcher', 'Creative_Writer', 'Analyzer', 'Verifier']:
        model_id = config.get(role_name, '')
        model_info = get_model_info(model_id)
        roles[role_name] = {
            'model_id': model_id,
            'model_name': model_info.get('name', model_id),
            'support_images': model_info.get('support_images', False)
        }
    return jsonify(roles)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all available models from model.json"""
    models = load_models()
    enabled = [m for m in models if m.get('enabled', True)]
    return jsonify(enabled)

@app.route('/api/upload_document', methods=['POST'])
def upload_document():
    """Handle PDF and image upload and extract text"""
    try:
        conversation_id = request.form.get('chat_id')
        if not conversation_id:
            return jsonify({'success': False, 'error': 'chat_id is required'})
        conv = get_conversation(conversation_id)

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        filename = secure_filename(file.filename)
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext not in ['pdf', 'jpg', 'jpeg', 'png']:
            return jsonify({'success': False, 'error': 'Only PDF, JPG, and PNG files are allowed'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        doc_payload = _extract_document_payload(filepath, filename)
        conv['uploaded_documents'][filename] = doc_payload

        return jsonify({
            'success': True,
            'document': _build_document_response(doc_payload)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/restore_documents', methods=['POST'])
def restore_documents():
    """Re-register previously uploaded documents from a saved chat"""
    try:
        data = request.json
        conversation_id = data.get('chat_id')
        if not conversation_id:
            return jsonify({'success': False, 'error': 'chat_id is required'})
        conv = get_conversation(conversation_id)
        documents = data.get('documents', [])
        restored = []

        for doc_meta in documents:
            filename = doc_meta.get('filename')
            if not filename:
                continue
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                continue

            doc_payload = _extract_document_payload(filepath, filename)
            conv['uploaded_documents'][filename] = doc_payload
            restored.append(_build_document_response(doc_payload))

        return jsonify({'success': True, 'documents': restored})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/remove_document', methods=['POST'])
def remove_document():
    """Remove uploaded document"""
    try:
        data = request.json
        conversation_id = data.get('chat_id')
        if not conversation_id:
            return jsonify({'success': False, 'error': 'chat_id is required'})
        conv = get_conversation(conversation_id)
        filename = data.get('filename')
        
        if filename in conv['uploaded_documents']:
            del conv['uploaded_documents'][filename]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'File not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_chat', methods=['POST'])
def save_chat():
    """Save a chat session to disk"""
    try:
        data = request.json
        chat_id = data.get('id')
        chat_name = data.get('name')
        messages = data.get('messages', [])
        selected_bots = data.get('selectedBots', [])
        
        chat_data = {
            'id': chat_id,
            'name': chat_name,
            'messages': messages,
            'selectedBots': selected_bots,
            'systemPrompt': data.get('systemPrompt', ''),
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'uploadedDocuments': data.get('uploadedDocuments', [])
        }
        
        # Save to file
        filepath = os.path.join(app.config['CHAT_HISTORY_FOLDER'], f'{chat_id}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'id': chat_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/list_chats', methods=['GET'])
def list_chats():
    """List all saved chat sessions"""
    try:
        chat_files = [f for f in os.listdir(app.config['CHAT_HISTORY_FOLDER']) if f.endswith('.json')]
        chats = []
        
        for filename in chat_files:
            filepath = os.path.join(app.config['CHAT_HISTORY_FOLDER'], filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                    chats.append({
                        'id': chat_data.get('id'),
                        'name': chat_data.get('name'),
                        'timestamp': chat_data.get('timestamp'),
                        'messageCount': len(chat_data.get('messages', [])),
                        'lastPreview': ((chat_data.get('messages', [])[-1].get('raw_content')
                                        if chat_data.get('messages', []) else '') or '')[:120],
                        'isGenerating': get_conversation(chat_data.get('id')).get('is_generating', False)
                    })
            except Exception as e:
                print(f"Error loading chat file {filename}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        chats.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'success': True, 'chats': chats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'chats': []})

@app.route('/api/load_chat/<chat_id>', methods=['GET'])
def load_chat(chat_id):
    """Load a specific chat session"""
    try:
        filepath = os.path.join(app.config['CHAT_HISTORY_FOLDER'], f'{chat_id}.json')
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Chat not found'})
        
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        if 'schema_version' not in chat_data:
            chat_data['schema_version'] = 1

        live_conv = conversations.get(chat_id)
        if live_conv and (live_conv.get('messages') or live_conv.get('is_generating')):
            chat_data['messages'] = _history_to_ui_messages(chat_id)
            chat_data['uploadedDocuments'] = _conversation_documents_to_ui(chat_id)

        chat_data['isGenerating'] = get_conversation(chat_id).get('is_generating', False)
        
        return jsonify({'success': True, 'chat': chat_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_chat', methods=['POST'])
def delete_chat():
    """Delete a saved chat session"""
    try:
        data = request.json
        chat_id = data.get('id')
        
        filepath = os.path.join(app.config['CHAT_HISTORY_FOLDER'], f'{chat_id}.json')
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Chat not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_temp_chat', methods=['POST'])
def save_temp_chat():
    """Save a temporary chat session that will be auto-deleted next day"""
    try:
        data = request.json
        chat_id = data.get('id', 'temp_' + str(int(time.time() * 1000)))
        messages = data.get('messages', [])
        selected_bots = data.get('selectedBots', [])
        uploaded_documents = data.get('uploadedDocuments', [])
        
        chat_data = {
            'id': chat_id,
            'name': 'Temporary Chat',
            'messages': messages,
            'selectedBots': selected_bots,
            'systemPrompt': data.get('systemPrompt', ''),
            'timestamp': datetime.now().isoformat(),
            'isTemporary': True,
            'uploadedDocuments': uploaded_documents
        }
        
        # Save to temp folder
        filepath = os.path.join(app.config['TEMP_CHAT_HISTORY_FOLDER'], f'{chat_id}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'id': chat_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/list_temp_chats', methods=['GET'])
def list_temp_chats():
    """List all temporary chat sessions from today"""
    try:
        temp_folder = app.config['TEMP_CHAT_HISTORY_FOLDER']
        chat_files = [f for f in os.listdir(temp_folder) if f.endswith('.json')]
        chats = []
        
        for filename in chat_files:
            filepath = os.path.join(temp_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                    chats.append({
                        'id': chat_data.get('id'),
                        'name': 'Temp Chat',
                        'timestamp': chat_data.get('timestamp'),
                        'messageCount': len(chat_data.get('messages', [])),
                        'isTemporary': True,
                        'lastPreview': ((chat_data.get('messages', [])[-1].get('raw_content')
                                        if chat_data.get('messages', []) else '') or '')[:120],
                        'isGenerating': get_conversation(chat_data.get('id')).get('is_generating', False)
                    })
            except Exception as e:
                print(f"Error loading temp chat file {filename}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        chats.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'success': True, 'chats': chats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'chats': []})

@app.route('/api/load_temp_chat/<chat_id>', methods=['GET'])
def load_temp_chat(chat_id):
    """Load a specific temporary chat session"""
    try:
        filepath = os.path.join(app.config['TEMP_CHAT_HISTORY_FOLDER'], f'{chat_id}.json')
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Temporary chat not found'})
        
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        if 'schema_version' not in chat_data:
            chat_data['schema_version'] = 1

        live_conv = conversations.get(chat_id)
        if live_conv and (live_conv.get('messages') or live_conv.get('is_generating')):
            chat_data['messages'] = _history_to_ui_messages(chat_id)
            chat_data['uploadedDocuments'] = _conversation_documents_to_ui(chat_id)

        chat_data['isGenerating'] = get_conversation(chat_id).get('is_generating', False)
        
        return jsonify({'success': True, 'chat': chat_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_temp_chat', methods=['POST'])
def delete_temp_chat():
    """Delete a temporary chat session"""
    try:
        data = request.json
        chat_id = data.get('id')
        
        filepath = os.path.join(app.config['TEMP_CHAT_HISTORY_FOLDER'], f'{chat_id}.json')
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Temporary chat not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def extract_pdf_text(filepath):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    return text

def encode_image_to_base64(filepath):
    """Convert image file to base64 data URL"""
    try:
        with open(filepath, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            # Detect image format from file extension
            ext = filepath.lower().split('.')[-1]
            mime_type = f"image/{ext}" if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp'] else "image/jpeg"
            if ext == 'jpg':
                mime_type = "image/jpeg"
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        return None

def extract_json_from_text(text):
    """Extract JSON object from text that might contain markdown code blocks or extra text"""
    import re
    # Try to find JSON in code blocks first
    match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\n?\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find raw JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def build_document_context(conversation_id, system_prompt, support_images):
    """Build document context and image URLs for a council role"""
    conv = get_conversation(conversation_id)
    documents = conv.get('uploaded_documents', {})
    image_urls = []
    if documents:
        if support_images:
            pdf_context = ""
            for filename, doc_info in documents.items():
                if doc_info['type'] == 'image':
                    img_base64 = encode_image_to_base64(doc_info['filepath'])
                    if img_base64:
                        image_urls.append(img_base64)
                else:
                    if not pdf_context:
                        pdf_context = "\n\n===== AVAILABLE DOCUMENTS =====\n"
                    pdf_context += f"\n--- PDF Document: {filename} ---\n"
                    pdf_context += doc_info['content'][:5000]
                    if len(doc_info['content']) > 5000:
                        pdf_context += "\n[... content truncated ...]"
                    pdf_context += "\n"
            if pdf_context:
                system_prompt += pdf_context
        else:
            pdf_context = "\n\n===== AVAILABLE DOCUMENTS =====\n"
            for filename, doc_info in documents.items():
                doc_type = "PDF Document" if doc_info['type'] == 'pdf' else "Image"
                pdf_context += f"\n--- {doc_type}: {filename} ---\n"
                pdf_context += doc_info['content'][:5000]
                if len(doc_info['content']) > 5000:
                    pdf_context += "\n[... content truncated ...]"
                pdf_context += "\n"
            system_prompt += pdf_context
    return system_prompt, image_urls

def _history_to_ui_messages(conversation_id):
    """Convert backend conversation history to UI-friendly message objects."""
    import re

    ui_messages = []
    conv = get_conversation(conversation_id)
    for msg in conv.get('messages', []):
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'user':
            ui_messages.append({
                'type': 'user',
                'sender': 'You',
                'content': content,
                'id': msg.get('id')
            })
            continue

        sender = msg.get('bot_name', 'AI')
        body = content

        # Parse legacy format: [Sender (id)] content
        match = re.match(r'^\[(.+?)\]\s*(.*)$', content, re.DOTALL)
        if match:
            sender = match.group(1).strip() or sender
            body = match.group(2)

        ui_messages.append({
            'type': 'ai',
            'sender': sender,
            'content': body,
            'raw_content': body,
            'id': msg.get('id'),
            'run_group_id': msg.get('run_group_id'),
            'run_id': msg.get('run_id') or msg.get('id'),
            'model_id': msg.get('model_id'),
            'role_name': msg.get('role_name'),
            'thinking': msg.get('thinking', ''),
            'stream_status': msg.get('stream_status', ''),
            'is_final_response': bool(msg.get('is_final_response', False))
        })

    return ui_messages

def persist_chat_snapshot(session_id, user_system_prompt=''):
    """Persist the current in-memory session to saved or temp storage."""
    try:
        with persistence_lock:
            conv = get_conversation(session_id)
            is_saved = os.path.exists(
                os.path.join(app.config['CHAT_HISTORY_FOLDER'], f'{session_id}.json')
            )
            folder = app.config['CHAT_HISTORY_FOLDER'] if is_saved else app.config['TEMP_CHAT_HISTORY_FOLDER']
            filepath = os.path.join(folder, f'{session_id}.json')

            chat_data = {}
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)

            chat_data['id'] = session_id
            chat_data['schema_version'] = 2
            if 'name' not in chat_data:
                chat_data['name'] = 'Saved Chat' if is_saved else 'Temporary Chat'
            chat_data['messages'] = _history_to_ui_messages(session_id)
            chat_data['selectedBots'] = chat_data.get('selectedBots', [])
            chat_data['systemPrompt'] = chat_data.get('systemPrompt', user_system_prompt)
            chat_data['timestamp'] = chat_data.get('timestamp', datetime.now().isoformat())
            chat_data['uploadedDocuments'] = [
                {
                    'filename': doc.get('filename'),
                    'type': doc.get('type'),
                    'size': doc.get('size'),
                    'pages': doc.get('pages'),
                    'width': doc.get('width'),
                    'height': doc.get('height')
                }
                for doc in conv.get('uploaded_documents', {}).values()
            ]
            if not is_saved:
                chat_data['isTemporary'] = True

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Snapshot save error for {session_id}: {e}")


def _conversation_documents_to_ui(conversation_id):
    """Convert in-memory uploaded document metadata into UI payload format."""
    conv = get_conversation(conversation_id)
    return [
        {
            'filename': doc.get('filename'),
            'type': doc.get('type'),
            'size': doc.get('size'),
            'pages': doc.get('pages'),
            'width': doc.get('width'),
            'height': doc.get('height')
        }
        for doc in conv.get('uploaded_documents', {}).values()
    ]

def run_council_role(role_name, role_label, model_id, system_prompt, user_prompt, chat_history, conversation_id, run_group_id, support_images=False, on_stream_progress=None, _retry=False):
    def emit_chat(event, payload=None):
        enriched = dict(payload or {})
        enriched['chat_id'] = conversation_id
        socketio.emit(event, enriched, to=conversation_id)

    """Run a single council role and stream its response"""
    try:
        conv = get_conversation(conversation_id)

        # Build document context
        system_prompt, image_urls = build_document_context(conversation_id, system_prompt, support_images)

        # Use a unique bot_id for the streaming UI
        bot_id = f"council-{role_name.lower()}-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().strftime("%H:%M:%S")
        is_final_response = ('final response' in role_label.lower())

        pending_message_id = uuid.uuid4().hex
        append_conversation_message(
            conversation_id,
            role='assistant',
            content=f"[{role_label} ({model_id})] ",
            id=pending_message_id,
            bot_name=role_label,
            bot_id=bot_id,
            run_group_id=run_group_id,
            run_id=pending_message_id,
            role_name=role_name,
            model_id=model_id,
            thinking='',
            stream_status='running',
            is_final_response=is_final_response
        )
        conv['pending_message_id'] = pending_message_id

        # Emit start event
        emit_chat('ai_response_start', {
            'bot_name': role_label,
            'bot_id': bot_id,
            'timestamp': timestamp,
            'message_id': pending_message_id,
            'run_id': pending_message_id,
            'run_group_id': run_group_id,
            'role_name': role_name,
            'model_id': model_id,
            'is_final_response': is_final_response
        })

        # Emit running status
        emit_chat('council_status', {
            'role': role_name,
            'status': 'running',
            'run_group_id': run_group_id
        })

        full_response = ''
        full_thinking = ''
        stopped_early = False
        last_snapshot_ts = 0.0

        for chunk_type, chunk in completion_response_stream(
            model=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            chat_history=chat_history,
            temperature=0.7,
            image_urls=image_urls if support_images else None
        ):
            if not should_continue_streaming(conversation_id, pending_message_id):
                stopped_early = True
                break

            if chunk_type == 'thinking':
                full_thinking += chunk
                emit_chat('ai_thinking_chunk', {
                    'bot_id': bot_id,
                    'chunk': chunk,
                    'message_id': pending_message_id,
                    'run_id': pending_message_id,
                    'run_group_id': run_group_id,
                    'role_name': role_name,
                    'model_id': model_id,
                    'is_final_response': is_final_response
                })
            else:
                full_response += chunk
                update_message_content(
                    conversation_id,
                    pending_message_id,
                    f"[{role_label} ({model_id})] {full_response}"
                )
                emit_chat('ai_response_chunk', {
                    'bot_id': bot_id,
                    'chunk': chunk,
                    'message_id': pending_message_id,
                    'run_id': pending_message_id,
                    'run_group_id': run_group_id,
                    'role_name': role_name,
                    'model_id': model_id,
                    'is_final_response': is_final_response
                })

                now = time.time()
                if on_stream_progress and (now - last_snapshot_ts) >= 1.0:
                    on_stream_progress()
                    last_snapshot_ts = now

        if stopped_early:
            timestamp = datetime.now().strftime("%H:%M:%S")
            emit_chat('console_log', {
                'message': f"[{timestamp}] {role_label} stopped by user"
            })
            partial_converted = convert_to_traditional_chinese(full_response)
            thinking_converted = convert_to_traditional_chinese(full_thinking) if full_thinking else ''
            emit_chat('ai_response_end', {
                'bot_name': role_label,
                'bot_id': bot_id,
                'message': partial_converted,
                'thinking': thinking_converted,
                'timestamp': timestamp,
                'stopped': True,
                'message_id': pending_message_id,
                'run_id': pending_message_id,
                'run_group_id': run_group_id,
                'role_name': role_name,
                'model_id': model_id,
                'is_final_response': is_final_response
            })
            emit_chat('council_status', {
                'role': role_name,
                'status': 'stopped',
                'run_group_id': run_group_id
            })
            update_message_content(conversation_id, pending_message_id, f"[{role_label} ({model_id})] {partial_converted}")
            update_message_fields(
                conversation_id,
                pending_message_id,
                thinking=thinking_converted,
                stream_status='stopped'
            )
            if conv.get('pending_message_id') == pending_message_id:
                conv['pending_message_id'] = None
            if on_stream_progress:
                on_stream_progress()
            return ''

        # Clean up self-references
        import re
        full_response = re.sub(r'^\s*\[.*?\]\s*', '', full_response.strip())

        # Convert to Traditional Chinese
        converted_response = convert_to_traditional_chinese(full_response)

        # Finalize the in-progress entry content.
        update_message_content(conversation_id, pending_message_id, f"[{role_label} ({model_id})] {converted_response}")

        # Log completion
        timestamp = datetime.now().strftime("%H:%M:%S")
        emit_chat('console_log', {'message': f"[{timestamp}] {role_label} completed"})

        # Finalize streaming bubble
        thinking_converted = convert_to_traditional_chinese(full_thinking) if full_thinking else ''
        emit_chat('ai_response_end', {
            'bot_name': role_label,
            'bot_id': bot_id,
            'message': converted_response,
            'thinking': thinking_converted,
            'timestamp': timestamp,
            'message_id': pending_message_id,
            'run_id': pending_message_id,
            'run_group_id': run_group_id,
            'role_name': role_name,
            'model_id': model_id,
            'is_final_response': is_final_response
        })

        update_message_fields(
            conversation_id,
            pending_message_id,
            thinking=thinking_converted,
            stream_status='done'
        )

        # Status done
        emit_chat('council_status', {
            'role': role_name,
            'status': 'done',
            'run_group_id': run_group_id
        })

        if on_stream_progress:
            on_stream_progress()

        if conv.get('pending_message_id') == pending_message_id:
            conv['pending_message_id'] = None

        return converted_response
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        emit_chat('console_log', {
            'message': f"[{timestamp}] {role_label} encountered an error: {str(e)}{', retrying...' if not _retry else ', giving up.'}"
        })
        if not _retry:
            return run_council_role(
                role_name, role_label, model_id, system_prompt, user_prompt,
                chat_history, conversation_id, run_group_id, support_images, on_stream_progress, _retry=True
            )
        # Second failure — skip this role
        notice = f"{role_label} has encountered an error and has to go without it."
        emit_chat('council_status', {'role': role_name, 'status': 'error'})
        bot_id = f"council-{role_name.lower()}"
        emit_chat('ai_response', {
            'bot_name': role_label,
            'bot_id': bot_id,
            'message': f"⚠️ {notice}",
            'timestamp': timestamp
        })
        return None


def _run_optional_council_role(role_name, task_key, base_prompt, user_message, user_system_prompt,
                               tasks, config, council_results, chat_history, conversation_id,
                               run_group_id, auto_save_chat, emit_chat, extra_context=''):
    """Execute one optional council role with consistent skip/log/run behavior."""
    role_task = tasks.get(task_key, 'SKIP')
    if role_task.upper().strip() == 'SKIP':
        emit_chat('council_status', {'role': role_name, 'status': 'skipped'})
        emit_chat('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] {role_name.replace('_', ' ')}: SKIPPED"
        })
        return None

    model_id = config.get(role_name, '')
    role_info = get_model_info(model_id)
    timestamp = datetime.now().strftime("%H:%M:%S")
    action_verb = 'reviewing team results' if role_name == 'Verifier' else 'working'
    emit_chat('console_log', {
        'message': f"[{timestamp}] {role_name.replace('_', ' ')} ({model_id}) {action_verb}..."
    })

    role_system_prompt = base_prompt + f"\n\nOriginal user request: {user_message}"
    if extra_context:
        role_system_prompt += extra_context
    if user_system_prompt:
        role_system_prompt = user_system_prompt + "\n\n" + role_system_prompt

    role_response = run_council_role(
        role_name=role_name,
        role_label=f"{role_name.replace('_', ' ')} ({model_id})",
        model_id=model_id,
        system_prompt=role_system_prompt,
        user_prompt=role_task,
        chat_history=chat_history,
        conversation_id=conversation_id,
        run_group_id=run_group_id,
        support_images=role_info.get('support_images', False),
        on_stream_progress=auto_save_chat
    )
    if role_response is not None:
        council_results[role_name] = role_response
        auto_save_chat()
    return role_response

@socketio.on('join_chat')
def handle_join_chat(data):
    chat_id = data.get('chat_id')
    if chat_id:
        join_room(chat_id)

@socketio.on('send_message')
def handle_message_wrapper(data):
    chat_id = data.get('chat_id')
    conversation_id = chat_id if chat_id else request.sid
    conv = get_conversation(conversation_id)
    if conv.get('is_generating'):
        emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Generation already running for this chat",
            'chat_id': conversation_id
        }, to=conversation_id)
        return

    conv['abort_event'].clear()
    conv['is_generating'] = True
    task = socketio.start_background_task(handle_message_task, data, conversation_id)
    conv['active_stream_task'] = task


def handle_message_task(data, conversation_id):
    """Handle user message and run the council workflow"""
    user_message = data.get('message', '')
    user_system_prompt = data.get('system_prompt', '')
    
    def emit_chat(event, payload=None):
        enriched = dict(payload or {})
        enriched['chat_id'] = conversation_id
        socketio.emit(event, enriched, to=conversation_id)
            
    def auto_save_chat():
        persist_chat_snapshot(conversation_id, user_system_prompt)

    def finalize_generation():
        conv['is_generating'] = False
        conv['active_stream_task'] = None
        conv['pending_message_id'] = None

    def stop_if_aborted():
        if conv['abort_event'].is_set():
            emit_chat('all_done')
            finalize_generation()
            return True
        return False

    conv = get_conversation(conversation_id)
    conv['run_group_counter'] = conv.get('run_group_counter', 0) + 1
    run_group_id = f"rungrp-{conv['run_group_counter']:06d}"
    conv['current_run_group_id'] = run_group_id
    append_conversation_message(conversation_id, "user", user_message)
    auto_save_chat()

    emit_chat('run_group_start', {
        'run_group_id': run_group_id,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })

    # Build chat history (exclude current message)
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conv['messages'][:-1]
    ]

    # Load council config
    config = load_config()
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit_chat('console_log', {
        'message': f"[{timestamp}] Council workflow started"
    })

    # Set all roles to waiting
    for role in COUNCIL_ROLES:
        emit_chat('council_status', {'role': role, 'status': 'waiting'})

    # ── Step 1: Leader distributes tasks ──
    leader_model_id = config.get('Leader', '')
    leader_info = get_model_info(leader_model_id)
    leader_sys = LEADER_DISTRIBUTE_PROMPT
    if user_system_prompt:
        leader_sys = user_system_prompt + "\n\n" + leader_sys

    emit_chat('console_log', {
        'message': f"[{timestamp}] Leader ({leader_model_id}) analyzing and distributing tasks..."
    })

    task_distribution_raw = run_council_role(
        role_name='Leader',
        role_label=f'Leader - Task Distribution ({leader_model_id})',
        model_id=leader_model_id,
        system_prompt=leader_sys,
        user_prompt=user_message,
        chat_history=chat_history,
        conversation_id=conversation_id,
        run_group_id=run_group_id,
        support_images=leader_info.get('support_images', False),
        on_stream_progress=auto_save_chat
    )

    if stop_if_aborted():
        return

    # Parse task distribution — abort if leader failed or returned invalid JSON
    if task_distribution_raw is None:
        emit_chat('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Leader failed to respond. Aborting council workflow."
        })
        emit_chat('all_done')
        finalize_generation()
        return

    try:
        tasks = json.loads(extract_json_from_text(task_distribution_raw))
    except (json.JSONDecodeError, Exception) as e:
        emit_chat('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Failed to parse task distribution: {str(e)}. Aborting council workflow."
        })
        emit_chat('all_done')
        finalize_generation()
        return

    council_results = {}

    # ── Step 2: Researcher ──
    _run_optional_council_role(
        role_name='Researcher',
        task_key='researcher_task',
        base_prompt=RESEARCHER_PROMPT,
        user_message=user_message,
        user_system_prompt=user_system_prompt,
        tasks=tasks,
        config=config,
        council_results=council_results,
        chat_history=chat_history,
        conversation_id=conversation_id,
        run_group_id=run_group_id,
        auto_save_chat=auto_save_chat,
        emit_chat=emit_chat
    )
    if stop_if_aborted():
        return

    # ── Step 3: Creative Writer ──
    creative_context = ""
    if 'Researcher' in council_results:
        creative_context = f"\n\n===== Researcher's Findings (for your context) =====\n{council_results['Researcher']}"
    _run_optional_council_role(
        role_name='Creative_Writer',
        task_key='creative_writer_task',
        base_prompt=CREATIVE_WRITER_PROMPT,
        user_message=user_message,
        user_system_prompt=user_system_prompt,
        tasks=tasks,
        config=config,
        council_results=council_results,
        chat_history=chat_history,
        conversation_id=conversation_id,
        run_group_id=run_group_id,
        auto_save_chat=auto_save_chat,
        emit_chat=emit_chat,
        extra_context=creative_context
    )
    if stop_if_aborted():
        return

    # ── Step 4: Analyzer ──
    _run_optional_council_role(
        role_name='Analyzer',
        task_key='analyzer_task',
        base_prompt=ANALYZER_PROMPT,
        user_message=user_message,
        user_system_prompt=user_system_prompt,
        tasks=tasks,
        config=config,
        council_results=council_results,
        chat_history=chat_history,
        conversation_id=conversation_id,
        run_group_id=run_group_id,
        auto_save_chat=auto_save_chat,
        emit_chat=emit_chat
    )
    if stop_if_aborted():
        return

    # ── Step 5: Verifier ──
    verifier_context = ""
    if 'Researcher' in council_results:
        verifier_context += f"\n\n===== Researcher's Findings =====\n{council_results['Researcher']}"
    if 'Creative_Writer' in council_results:
        verifier_context += f"\n\n===== Creative Writer's Output =====\n{council_results['Creative_Writer']}"
    if 'Analyzer' in council_results:
        verifier_context += f"\n\n===== Analyzer's Calculations =====\n{council_results['Analyzer']}"
    _run_optional_council_role(
        role_name='Verifier',
        task_key='verifier_task',
        base_prompt=VERIFIER_PROMPT,
        user_message=user_message,
        user_system_prompt=user_system_prompt,
        tasks=tasks,
        config=config,
        council_results=council_results,
        chat_history=chat_history,
        conversation_id=conversation_id,
        run_group_id=run_group_id,
        auto_save_chat=auto_save_chat,
        emit_chat=emit_chat,
        extra_context=f"\n\nTeam outputs to review:{verifier_context}" if verifier_context else ''
    )
    if stop_if_aborted():
        return

    # ── Step 6: Leader synthesizes all results ──
    if council_results:
        timestamp = datetime.now().strftime("%H:%M:%S")
        emit_chat('console_log', {
            'message': f"[{timestamp}] Leader ({leader_model_id}) combining results..."
        })

        ordered_keys = ['Researcher', 'Creative_Writer', 'Analyzer', 'Verifier']
        results_text = ""
        for key in ordered_keys:
            if key in council_results:
                label = key.replace('_', ' ')
                results_text += f"\n\n===== {label} =====\n{council_results[key]}"

        combine_sys = LEADER_COMBINE_PROMPT + f"\n\nTeam outputs:{results_text}"
        if user_system_prompt:
            combine_sys = user_system_prompt + "\n\n" + combine_sys

        run_council_role(
            role_name='Leader',
            role_label=f'Leader - Final Response ({leader_model_id})',
            model_id=leader_model_id,
            system_prompt=combine_sys,
            user_prompt=f"Please combine the team's work and provide the final response to: {user_message}",
            chat_history=chat_history,
            conversation_id=conversation_id,
            run_group_id=run_group_id,
            support_images=leader_info.get('support_images', False),
            on_stream_progress=auto_save_chat
        )
        auto_save_chat()

    # Signal all done
    emit_chat('all_done')
    emit_chat('run_group_end', {
        'run_group_id': run_group_id,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    conv['current_run_group_id'] = None
    finalize_generation()

@socketio.on('stop_generation')
def handle_stop(data):
    """Handle stop generation request"""
    conversation_id = data.get('chat_id') if data and data.get('chat_id') else request.sid
    conv = get_conversation(conversation_id)
    conv['abort_event'].set()
    conv['is_generating'] = False
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit('console_log', {
        'message': f"[{timestamp}] Stop requested - will stop after current AI finishes",
        'chat_id': conversation_id
    }, to=conversation_id)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('console_log', {'message': f"[{datetime.now().strftime('%H:%M:%S')}] Connected to server"})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    # Keep per-chat state alive after UI switches/disconnects.
    # Conversation lifecycle is tied to chat IDs and persisted history, not socket IDs.
    print('Client disconnected')

@socketio.on('clear_history')
def handle_clear_history(data=None):
    """Clear conversation history for the current session"""
    chat_id = data.get('chat_id') if data else None
    conversation_id = chat_id if chat_id else request.sid
    conv = get_conversation(conversation_id)
    conv['messages'] = []
    conv['uploaded_documents'] = {}
    conv['pending_message_id'] = None
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit('console_log', {'message': f"[{timestamp}] Conversation history cleared", 'chat_id': conversation_id}, to=conversation_id)

@socketio.on('load_chat_history')
def handle_load_chat_history(data):
    """Update backend conversation history when a chat is loaded"""
    chat_id = data.get('chat_id')
    conversation_id = chat_id if chat_id else request.sid
    conv = get_conversation(conversation_id)
    messages = data.get('messages', [])

    # Prevent stale client snapshots from replacing live in-memory generation state.
    if conv.get('is_generating'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        emit('console_log', {
            'message': f"[{timestamp}] Ignored chat history sync during active generation",
            'chat_id': conversation_id
        }, to=conversation_id)
        return
    
    # Convert UI message format to backend conversation history format
    conv['messages'] = []
    for msg in messages:
        if msg.get('type') == 'user':
            append_conversation_message(
                conversation_id,
                role='user',
                content=msg.get('content', ''),
                id=msg.get('id', uuid.uuid4().hex)
            )
        elif msg.get('type') == 'ai':
            # AI messages in the format [Bot Name (bot-id)] message
            bot_name = msg.get('botName', 'AI')
            bot_id = msg.get('botId', 'unknown')
            content = msg.get('content', '')
            append_conversation_message(
                conversation_id,
                role='assistant',
                content=f"[{bot_name} ({bot_id})] {content}",
                id=msg.get('id', uuid.uuid4().hex),
                bot_name=bot_name,
                bot_id=bot_id,
                run_group_id=msg.get('run_group_id'),
                run_id=msg.get('run_id') or msg.get('id'),
                model_id=msg.get('model_id'),
                role_name=msg.get('role_name'),
                thinking=msg.get('thinking', ''),
                stream_status=msg.get('stream_status', ''),
                is_final_response=bool(msg.get('is_final_response', False))
            )
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit('console_log', {
        'message': f"[{timestamp}] Chat history loaded ({len(messages)} messages)",
        'chat_id': conversation_id
    }, to=conversation_id)

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Visit http://localhost:5000 in your browser")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
