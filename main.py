from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import time
import threading
from datetime import datetime
from GPT_handle import completion_response_stream, convert_to_traditional_chinese
import os
from werkzeug.utils import secure_filename
import PyPDF2
from PIL import Image
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHAT_HISTORY_FOLDER'] = 'chat_history'
app.config['TEMP_CHAT_HISTORY_FOLDER'] = 'temp_chat_history'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")

# Store conversation history for each session
conversation_histories = {}
# Store uploaded PDFs content
pdf_documents = {}
# Track stop requests per session
stop_flags = {}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Create chat history directory if it doesn't exist
os.makedirs(app.config['CHAT_HISTORY_FOLDER'], exist_ok=True)
# Create temp chat history directory if it doesn't exist
os.makedirs(app.config['TEMP_CHAT_HISTORY_FOLDER'], exist_ok=True)

# Load council role configurations
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'Leader': 'gpt-4o',
            'Researcher': 'gpt-4o',
            'Creative_Writer': 'gpt-4o',
            'Analyzer': 'gpt-4o',
            'Verifier': 'gpt-4o'
        }

def load_models():
    """Load available models from model.json"""
    try:
        with open('model.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_model_info(model_id):
    """Get model details by ID from model.json"""
    models = load_models()
    for m in models:
        if m['id'] == model_id:
            return m
    return {'id': model_id, 'name': model_id, 'support_images': False}

# Council role system prompts
LEADER_DISTRIBUTE_PROMPT = """You are the Leader of an AI council. Your job is to analyze the user's request and distribute tasks to your team members.

Your team:
- Researcher: Gathers factual information, grounds responses with data, finds relevant facts and sources
- Creative_Writer: Writes creative content, generates engaging narratives, provides creative ideas and solutions
- Analyzer: Performs calculations, data analysis, and mathematical reasoning
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
        
        file_size = os.path.getsize(filepath)
        
        # Process based on file type
        if file_ext == 'pdf':
            text_content = extract_pdf_text(filepath)
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                page_count = len(pdf_reader.pages)
            
            pdf_documents[filename] = {
                'filename': filename,
                'content': text_content,
                'pages': page_count,
                'size': file_size,
                'type': 'pdf'
            }
            
            return jsonify({
                'success': True,
                'document': {
                    'filename': filename,
                    'pages': page_count,
                    'size': file_size,
                    'type': 'pdf'
                }
            })
        else:
            # Handle image files
            text_content = "[Image uploaded]"
            img = Image.open(filepath)
            width, height = img.size
            
            pdf_documents[filename] = {
                'filename': filename,
                'content': text_content,
                'filepath': filepath,  # Store the actual file path
                'width': width,
                'height': height,
                'size': file_size,
                'type': 'image'
            }
            
            return jsonify({
                'success': True,
                'document': {
                    'filename': filename,
                    'width': width,
                    'height': height,
                    'size': file_size,
                    'type': 'image'
                }
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/remove_document', methods=['POST'])
def remove_document():
    """Remove uploaded document"""
    try:
        data = request.json
        filename = data.get('filename')
        
        if filename in pdf_documents:
            del pdf_documents[filename]
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
            'timestamp': data.get('timestamp', datetime.now().isoformat())
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
                        'messageCount': len(chat_data.get('messages', []))
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
        
        chat_data = {
            'id': chat_id,
            'name': 'Temporary Chat',
            'messages': messages,
            'selectedBots': selected_bots,
            'systemPrompt': data.get('systemPrompt', ''),
            'timestamp': datetime.now().isoformat(),
            'isTemporary': True
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
                        'isTemporary': True
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

def build_document_context(system_prompt, support_images):
    """Build document context and image URLs for a council role"""
    image_urls = []
    if pdf_documents:
        if support_images:
            pdf_context = ""
            for filename, doc_info in pdf_documents.items():
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
            for filename, doc_info in pdf_documents.items():
                doc_type = "PDF Document" if doc_info['type'] == 'pdf' else "Image"
                pdf_context += f"\n--- {doc_type}: {filename} ---\n"
                pdf_context += doc_info['content'][:5000]
                if len(doc_info['content']) > 5000:
                    pdf_context += "\n[... content truncated ...]"
                pdf_context += "\n"
            system_prompt += pdf_context
    return system_prompt, image_urls

def run_council_role(role_name, role_label, model_id, system_prompt, user_prompt, chat_history, session_id, support_images=False, _retry=False):
    """Run a single council role and stream its response"""
    try:
        # Build document context
        system_prompt, image_urls = build_document_context(system_prompt, support_images)

        # Use a unique bot_id for the streaming UI
        bot_id = f"council-{role_name.lower()}"
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Emit start event
        socketio.emit('ai_response_start', {
            'bot_name': role_label,
            'bot_id': bot_id,
            'timestamp': timestamp
        })

        # Emit running status
        socketio.emit('council_status', {
            'role': role_name,
            'status': 'running'
        })

        # Stream the response
        full_response = ''
        full_thinking = ''
        stopped_early = False

        for chunk_type, chunk in completion_response_stream(
            model=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            chat_history=chat_history,
            temperature=0.7,
            image_urls=image_urls if support_images else None
        ):
            if stop_flags.get(session_id, False):
                stopped_early = True
                break
            if chunk_type == 'thinking':
                full_thinking += chunk
                socketio.emit('ai_thinking_chunk', {
                    'bot_id': bot_id,
                    'chunk': chunk
                })
            else:
                full_response += chunk
                socketio.emit('ai_response_chunk', {
                    'bot_id': bot_id,
                    'chunk': chunk
                })

        if stopped_early:
            timestamp = datetime.now().strftime("%H:%M:%S")
            socketio.emit('console_log', {
                'message': f"[{timestamp}] {role_label} stopped by user"
            })
            partial_converted = convert_to_traditional_chinese(full_response)
            thinking_converted = convert_to_traditional_chinese(full_thinking) if full_thinking else ''
            socketio.emit('ai_response_end', {
                'bot_name': role_label,
                'bot_id': bot_id,
                'message': partial_converted,
                'thinking': thinking_converted,
                'timestamp': timestamp,
                'stopped': True
            })
            socketio.emit('council_status', {
                'role': role_name,
                'status': 'stopped'
            })
            return ''

        # Clean up self-references
        import re
        full_response = re.sub(r'^\s*\[.*?\]\s*', '', full_response.strip())

        # Convert to Traditional Chinese
        converted_response = convert_to_traditional_chinese(full_response)

        # Store in conversation history
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []
        conversation_histories[session_id].append({
            "role": "assistant",
            "content": f"[{role_label} ({model_id})] {converted_response}",
            "bot_name": role_label,
            "bot_id": bot_id
        })

        # Log completion
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('console_log', {'message': f"[{timestamp}] {role_label} completed"})

        # Finalize streaming bubble
        thinking_converted = convert_to_traditional_chinese(full_thinking) if full_thinking else ''
        socketio.emit('ai_response_end', {
            'bot_name': role_label,
            'bot_id': bot_id,
            'message': converted_response,
            'thinking': thinking_converted,
            'timestamp': timestamp
        })

        # Status done
        socketio.emit('council_status', {
            'role': role_name,
            'status': 'done'
        })

        return converted_response
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('console_log', {
            'message': f"[{timestamp}] {role_label} encountered an error: {str(e)}{', retrying...' if not _retry else ', giving up.'}"
        })
        if not _retry:
            return run_council_role(
                role_name, role_label, model_id, system_prompt, user_prompt,
                chat_history, session_id, support_images, _retry=True
            )
        # Second failure — skip this role
        notice = f"{role_label} has encountered an error and has to go without it."
        socketio.emit('council_status', {'role': role_name, 'status': 'error'})
        bot_id = f"council-{role_name.lower()}"
        socketio.emit('ai_response', {
            'bot_name': role_label,
            'bot_id': bot_id,
            'message': f"⚠️ {notice}",
            'timestamp': timestamp
        })
        return None

@socketio.on('send_message')
def handle_message(data):
    """Handle user message and run the council workflow"""
    user_message = data.get('message', '')
    user_system_prompt = data.get('system_prompt', '')
    session_id = request.sid

    # Reset stop flag
    stop_flags[session_id] = False

    # Initialize conversation history
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    conversation_histories[session_id].append({
        "role": "user",
        "content": user_message
    })

    # Build chat history (exclude current message)
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_histories[session_id][:-1]
    ]

    # Load council config
    config = load_config()
    timestamp = datetime.now().strftime("%H:%M:%S")
    socketio.emit('console_log', {
        'message': f"[{timestamp}] Council workflow started"
    })

    # Set all roles to waiting
    for role in ['Leader', 'Researcher', 'Creative_Writer', 'Analyzer', 'Verifier']:
        socketio.emit('council_status', {'role': role, 'status': 'waiting'})

    # ── Step 1: Leader distributes tasks ──
    leader_model_id = config.get('Leader', '')
    leader_info = get_model_info(leader_model_id)
    leader_sys = LEADER_DISTRIBUTE_PROMPT
    if user_system_prompt:
        leader_sys = user_system_prompt + "\n\n" + leader_sys

    socketio.emit('console_log', {
        'message': f"[{timestamp}] Leader ({leader_model_id}) analyzing and distributing tasks..."
    })

    task_distribution_raw = run_council_role(
        role_name='Leader',
        role_label=f'Leader - Task Distribution ({leader_model_id})',
        model_id=leader_model_id,
        system_prompt=leader_sys,
        user_prompt=user_message,
        chat_history=chat_history,
        session_id=session_id,
        support_images=leader_info.get('support_images', False)
    )

    if stop_flags.get(session_id, False):
        socketio.emit('all_done')
        return

    # Parse task distribution — abort if leader failed or returned invalid JSON
    if task_distribution_raw is None:
        socketio.emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Leader failed to respond. Aborting council workflow."
        })
        socketio.emit('all_done')
        return

    try:
        tasks = json.loads(extract_json_from_text(task_distribution_raw))
    except (json.JSONDecodeError, Exception) as e:
        socketio.emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Failed to parse task distribution: {str(e)}. Aborting council workflow."
        })
        socketio.emit('all_done')
        return

    council_results = {}

    # ── Step 2: Researcher ──
    researcher_task = tasks.get('researcher_task', 'SKIP')
    if researcher_task.upper().strip() != 'SKIP':
        researcher_model_id = config.get('Researcher', '')
        researcher_info = get_model_info(researcher_model_id)
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('console_log', {
            'message': f"[{timestamp}] Researcher ({researcher_model_id}) working..."
        })

        researcher_sys = RESEARCHER_PROMPT + f"\n\nOriginal user request: {user_message}"
        if user_system_prompt:
            researcher_sys = user_system_prompt + "\n\n" + researcher_sys

        researcher_response = run_council_role(
            role_name='Researcher',
            role_label=f'Researcher ({researcher_model_id})',
            model_id=researcher_model_id,
            system_prompt=researcher_sys,
            user_prompt=researcher_task,
            chat_history=chat_history,
            session_id=session_id,
            support_images=researcher_info.get('support_images', False)
        )
        if researcher_response is not None:
            council_results['Researcher'] = researcher_response

        if stop_flags.get(session_id, False):
            socketio.emit('all_done')
            return
    else:
        socketio.emit('council_status', {'role': 'Researcher', 'status': 'skipped'})
        socketio.emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Researcher: SKIPPED"
        })

    # ── Step 3: Creative Writer ──
    creative_task = tasks.get('creative_writer_task', 'SKIP')
    if creative_task.upper().strip() != 'SKIP':
        creative_model_id = config.get('Creative_Writer', '')
        creative_info = get_model_info(creative_model_id)
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('console_log', {
            'message': f"[{timestamp}] Creative Writer ({creative_model_id}) working..."
        })

        creative_sys = CREATIVE_WRITER_PROMPT + f"\n\nOriginal user request: {user_message}"
        if 'Researcher' in council_results:
            creative_sys += f"\n\n===== Researcher's Findings (for your context) =====\n{council_results['Researcher']}"
        if user_system_prompt:
            creative_sys = user_system_prompt + "\n\n" + creative_sys

        creative_response = run_council_role(
            role_name='Creative_Writer',
            role_label=f'Creative Writer ({creative_model_id})',
            model_id=creative_model_id,
            system_prompt=creative_sys,
            user_prompt=creative_task,
            chat_history=chat_history,
            session_id=session_id,
            support_images=creative_info.get('support_images', False)
        )
        if creative_response is not None:
            council_results['Creative_Writer'] = creative_response

        if stop_flags.get(session_id, False):
            socketio.emit('all_done')
            return
    else:
        socketio.emit('council_status', {'role': 'Creative_Writer', 'status': 'skipped'})
        socketio.emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Creative Writer: SKIPPED"
        })

    # ── Step 4: Analyzer ──
    analyzer_task = tasks.get('analyzer_task', 'SKIP')
    if analyzer_task.upper().strip() != 'SKIP':
        analyzer_model_id = config.get('Analyzer', '')
        analyzer_info = get_model_info(analyzer_model_id)
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('console_log', {
            'message': f"[{timestamp}] Analyzer ({analyzer_model_id}) working..."
        })

        analyzer_sys = ANALYZER_PROMPT + f"\n\nOriginal user request: {user_message}"
        if user_system_prompt:
            analyzer_sys = user_system_prompt + "\n\n" + analyzer_sys

        analyzer_response = run_council_role(
            role_name='Analyzer',
            role_label=f'Analyzer ({analyzer_model_id})',
            model_id=analyzer_model_id,
            system_prompt=analyzer_sys,
            user_prompt=analyzer_task,
            chat_history=chat_history,
            session_id=session_id,
            support_images=analyzer_info.get('support_images', False)
        )
        if analyzer_response is not None:
            council_results['Analyzer'] = analyzer_response

        if stop_flags.get(session_id, False):
            socketio.emit('all_done')
            return
    else:
        socketio.emit('council_status', {'role': 'Analyzer', 'status': 'skipped'})
        socketio.emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Analyzer: SKIPPED"
        })

    # ── Step 5: Verifier ──
    verifier_task = tasks.get('verifier_task', 'SKIP')
    if verifier_task.upper().strip() != 'SKIP':
        verifier_model_id = config.get('Verifier', '')
        verifier_info = get_model_info(verifier_model_id)
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('console_log', {
            'message': f"[{timestamp}] Verifier ({verifier_model_id}) reviewing team results..."
        })

        # Build a full context block of all previous results for the Verifier
        verifier_context = ""
        if 'Researcher' in council_results:
            verifier_context += f"\n\n===== Researcher's Findings =====\n{council_results['Researcher']}"
        if 'Creative_Writer' in council_results:
            verifier_context += f"\n\n===== Creative Writer's Output =====\n{council_results['Creative_Writer']}"
        if 'Analyzer' in council_results:
            verifier_context += f"\n\n===== Analyzer's Calculations =====\n{council_results['Analyzer']}"

        verifier_sys = VERIFIER_PROMPT + f"\n\nOriginal user request: {user_message}"
        if verifier_context:
            verifier_sys += f"\n\nTeam outputs to review:{verifier_context}"
        if user_system_prompt:
            verifier_sys = user_system_prompt + "\n\n" + verifier_sys

        verifier_response = run_council_role(
            role_name='Verifier',
            role_label=f'Verifier ({verifier_model_id})',
            model_id=verifier_model_id,
            system_prompt=verifier_sys,
            user_prompt=verifier_task,
            chat_history=chat_history,
            session_id=session_id,
            support_images=verifier_info.get('support_images', False)
        )
        if verifier_response is not None:
            council_results['Verifier'] = verifier_response

        if stop_flags.get(session_id, False):
            socketio.emit('all_done')
            return
    else:
        socketio.emit('council_status', {'role': 'Verifier', 'status': 'skipped'})
        socketio.emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Verifier: SKIPPED"
        })

    # ── Step 6: Leader synthesizes all results ──
    if council_results:
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('console_log', {
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
            session_id=session_id,
            support_images=leader_info.get('support_images', False)
        )

    # Signal all done
    socketio.emit('all_done')

@socketio.on('stop_generation')
def handle_stop():
    """Handle stop generation request"""
    session_id = request.sid
    stop_flags[session_id] = True
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit('console_log', {'message': f"[{timestamp}] Stop requested - will stop after current AI finishes"})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('console_log', {'message': f"[{datetime.now().strftime('%H:%M:%S')}] Connected to server"})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    # Clean up stop flags
    if session_id in stop_flags:
        del stop_flags[session_id]
    print('Client disconnected')

@socketio.on('clear_history')
def handle_clear_history():
    """Clear conversation history for the current session"""
    session_id = request.sid
    if session_id in conversation_histories:
        conversation_histories[session_id] = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit('console_log', {'message': f"[{timestamp}] Conversation history cleared"})

@socketio.on('load_chat_history')
def handle_load_chat_history(data):
    """Update backend conversation history when a chat is loaded"""
    session_id = request.sid
    messages = data.get('messages', [])
    
    # Convert UI message format to backend conversation history format
    conversation_histories[session_id] = []
    for msg in messages:
        if msg.get('type') == 'user':
            conversation_histories[session_id].append({
                "role": "user",
                "content": msg.get('content', '')
            })
        elif msg.get('type') == 'ai':
            # AI messages in the format [Bot Name (bot-id)] message
            bot_name = msg.get('botName', 'AI')
            bot_id = msg.get('botId', 'unknown')
            content = msg.get('content', '')
            conversation_histories[session_id].append({
                "role": "assistant",
                "content": f"[{bot_name} ({bot_id})] {content}",
                "bot_name": bot_name,
                "bot_id": bot_id
            })
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit('console_log', {'message': f"[{timestamp}] Chat history loaded ({len(messages)} messages)"})

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Visit http://localhost:5000 in your browser")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
