from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import time
import threading
from datetime import datetime
from GPT_handle import completion_response, completion_response_stream, convert_to_traditional_chinese
import os
from werkzeug.utils import secure_filename
import PyPDF2
from PIL import Image
import easyocr
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

# Load bot configurations
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'bots': [], 'web_search_bot': 'grok-3-deepsearch'}

def load_bots():
    return load_config().get('bots', [])

def get_web_search_bot():
    return load_config().get('web_search_bot', 'grok-3-deepsearch')

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

@app.route('/api/bots', methods=['GET'])
def get_bots():
    """Get all available bots from config"""
    bots = load_bots()
    # Filter out disabled bots
    enabled_bots = [bot for bot in bots if bot.get('enabled', True)]
    return jsonify(enabled_bots)

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
            text_content = extract_image_text(filepath)
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

# Initialize EasyOCR reader (lazy loading)
_ocr_reader = None

def get_ocr_reader():
    """Get or initialize EasyOCR reader"""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False)  # Use CPU, English only
    return _ocr_reader

def perform_web_search_once(user_message):
    """Perform web search once using configured web search model"""
    try:
        # Get the configured web search bot
        web_search_bot = get_web_search_bot()
        
        # Use configured model to get web search results
        search_response = completion_response(
            model=web_search_bot,
            system_prompt="""**System Prompt:**
You are a web research assistant that gathers and delivers accurate, relevant information from the web to support other AI assistants.
**Core Instructions:**
- Respond only with the information you find. Do not add any commentary, opinions, introductions, summaries, conclusions, or meta-statements.
- Present the information in clear, structured markdown (headings, bullet points, tables, numbered lists).
- Include all key facts, data, statistics, quotes, dates, names, and details from credible sources.
- Cite sources inline where relevant (e.g., after a fact or paragraph) with the full URL, and list all sources at the end under a "Sources" heading.
- Prioritize recent, authoritative sources (official websites, reputable news, academic/government reports).
- If sources conflict, present both sides clearly with their respective citations.
- If no reliable information is found, state only: "No reliable information found on [topic] from checked sources."
Deliver only the structured factual content and citations.
""",
            user_prompt=user_message,
            chat_history=None,
            temperature=0.7
        )
        return search_response
    except Exception as e:
        return f"[Web Search Error: {str(e)}]"

def get_ai_search_query(bot_id, user_message, chat_history):
    """Ask an AI model what it wants to search for to answer the user's question"""
    try:
        query_response = completion_response(
            model=bot_id,
            system_prompt="You are deciding what to search for on the web to best answer the user's question. Respond with ONLY the search query string — no explanation, no quotes, no surrounding text. Just the bare search query.",
            user_prompt=f"What would you search for on the web to give the best answer to this message? User message: {user_message}",
            chat_history=chat_history[-5:] if chat_history else None,
            temperature=0.3
        )
        return query_response.strip()
    except Exception as e:
        return user_message  # Fall back to user message as query
def extract_image_text(filepath):
    """Extract text from image using OCR"""
    text = ""
    try:
        reader = get_ocr_reader()
        result = reader.readtext(filepath)
        # Combine all detected text
        text = "\n".join([item[1] for item in result])
        if not text.strip():
            text = "[Image contains no detectable text or is purely visual]"
    except Exception as e:
        text = f"[Image uploaded but text extraction failed: {str(e)}]"
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

def simulate_ai_response(bot_name, bot_id, user_message, selected_bots, session_id, chat_history, current_round_responses, web_search_results=None, support_images=False, individual_web_search=False, user_system_prompt=''):
    """Process AI response using GPT_handle"""
    try:
        # Build enhanced chat history including current round responses
        enhanced_history = chat_history.copy()
        
        # Add current round responses from agents that already responded
        if current_round_responses:
            for prev_response in current_round_responses:
                enhanced_history.append({
                    "role": "assistant",
                    "content": f"[{prev_response['bot_name']} ({prev_response['bot_id']})] {prev_response['message']}"
                })
        
        # Use GPT_handle to get actual AI response with enhanced chat history
        system_prompt = f"You are {bot_name} (model: {bot_id}), an AI assistant participating in a council discussion with other AI models. You can see responses from other AI assistants and should engage with their ideas, build upon them, agree, disagree, or add new perspectives. Be aware that you are communicating with different AI models, not the same assistant. Please use Markdown formatting including tables, lists, and code blocks where appropriate. Do NOT prefix your response with your name or model ID - the system will add that automatically."

        # Prepend user-defined system prompt if provided
        if user_system_prompt:
            system_prompt = user_system_prompt + "\n\n" + system_prompt
        
        # Add web search results if available
        if web_search_results:
            system_prompt += f"\n\n===== WEB SEARCH RESULTS =====\n{web_search_results}\n\nUse the above web search results to provide accurate and up-to-date information in your response."

        # Perform individual web search for this bot if enabled
        if individual_web_search:
            ts = datetime.now().strftime("%H:%M:%S")
            socketio.emit('individual_search_start', {
                'bot_id': bot_id,
                'bot_name': bot_name,
                'timestamp': ts
            })
            socketio.emit('console_log', {
                'message': f"[{ts}] {bot_name}: determining individual search query..."
            })
            # Ask the bot what it wants to search for
            search_query = get_ai_search_query(bot_id, user_message, enhanced_history)
            socketio.emit('console_log', {
                'message': f"[{ts}] {bot_name} searching for: {search_query}"
            })
            # Perform the search
            individual_results = perform_web_search_once(search_query)
            # Emit results — private to this bot, visible only to user
            converted_individual_results = convert_to_traditional_chinese(individual_results)
            ts = datetime.now().strftime("%H:%M:%S")
            socketio.emit('individual_search_results', {
                'bot_id': bot_id,
                'bot_name': bot_name,
                'query': search_query,
                'results': converted_individual_results,
                'timestamp': ts
            })
            socketio.emit('console_log', {
                'message': f"[{ts}] {bot_name} individual search completed"
            })
            # Inject results into THIS bot's system prompt only (not shared)
            system_prompt += f"\n\n===== YOUR INDIVIDUAL WEB SEARCH RESULTS =====\nYou chose to search for: \"{search_query}\"\n{individual_results}\n\nUse these search results — which only you have access to — to inform your response."

        # Prepare image URLs for models that support images
        image_urls = []
        
        # Add PDF/Image context
        if pdf_documents:
            if support_images:
                # For models that support images, send images directly and PDFs as text
                pdf_context = ""
                for filename, doc_info in pdf_documents.items():
                    if doc_info['type'] == 'image':
                        # Encode and add image to image_urls list
                        img_base64 = encode_image_to_base64(doc_info['filepath'])
                        if img_base64:
                            image_urls.append(img_base64)
                    else:
                        # Add PDF content as text
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
                # For models that don't support images, add everything as text (OCR extracted)
                pdf_context = "\n\n===== AVAILABLE DOCUMENTS =====\n"
                for filename, doc_info in pdf_documents.items():
                    doc_type = "PDF Document" if doc_info['type'] == 'pdf' else "Image"
                    pdf_context += f"\n--- {doc_type}: {filename} ---\n"
                    pdf_context += doc_info['content'][:5000]  # Limit to first 5000 chars per document
                    if len(doc_info['content']) > 5000:
                        pdf_context += "\n[... content truncated ...]"
                    pdf_context += "\n"
                system_prompt += pdf_context
        
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Emit start event so the frontend can create the message bubble
        socketio.emit('ai_response_start', {
            'bot_name': bot_name,
            'bot_id': bot_id,
            'timestamp': timestamp
        })

        # Stream the response chunk by chunk
        full_response = ''
        full_thinking = ''
        stopped_early = False

        for chunk_type, chunk in completion_response_stream(
            model=bot_id,
            system_prompt=system_prompt,
            user_prompt=user_message,
            chat_history=enhanced_history,
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
                'message': f"[{timestamp}] {bot_name} response stopped by user"
            })
            # Finalize the bubble with whatever was streamed so far
            partial_converted = convert_to_traditional_chinese(full_response)
            thinking_converted = convert_to_traditional_chinese(full_thinking) if full_thinking else ''
            socketio.emit('ai_response_end', {
                'bot_name': bot_name,
                'bot_id': bot_id,
                'message': partial_converted,
                'thinking': thinking_converted,
                'timestamp': timestamp,
                'stopped': True
            })
            socketio.emit('ai_status', {
                'bot_id': bot_id,
                'status': 'stopped'
            })
            return ''

        # Clean up self-references if AI includes them despite instructions
        import re
        # Remove patterns like [Bot Name (model-id)] or [Bot Name] at the start
        full_response = re.sub(r'^\s*\[.*?\]\s*', '', full_response.strip())

        # Convert full response to Traditional Chinese
        converted_response = convert_to_traditional_chinese(full_response)

        # Store AI response in conversation history (using converted version)
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []
        conversation_histories[session_id].append({
            "role": "assistant",
            "content": f"[{bot_name} ({bot_id})] {converted_response}",
            "bot_name": bot_name,
            "bot_id": bot_id
        })

        # Log completion to console
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('console_log', {'message': f"[{timestamp}] {bot_name} completed processing"})

        # Finalize the streaming bubble with the fully converted/formatted response
        thinking_converted = convert_to_traditional_chinese(full_thinking) if full_thinking else ''
        socketio.emit('ai_response_end', {
            'bot_name': bot_name,
            'bot_id': bot_id,
            'message': converted_response,
            'thinking': thinking_converted,
            'timestamp': timestamp
        })

        # Emit 'done' status
        socketio.emit('ai_status', {
            'bot_id': bot_id,
            'status': 'done'
        })

        return converted_response
    except Exception as e:
        # Log error to console
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_message = f"[{timestamp}] {bot_name} encountered an error: {str(e)}"
        socketio.emit('console_log', {'message': error_message})
        
        # Emit 'error' status
        socketio.emit('ai_status', {
            'bot_id': bot_id,
            'status': 'error'
        })
        
        # Send error response to conversation window
        error_msg = f"[Error] {bot_name} could not process the request: {str(e)}"
        converted_error = convert_to_traditional_chinese(error_msg)
        socketio.emit('ai_response', {
            'bot_name': bot_name,
            'bot_id': bot_id,
            'message': converted_error,
            'timestamp': timestamp
        })
        
        return error_msg

@socketio.on('send_message')
def handle_message(data):
    """Handle user message and distribute to selected AIs"""
    user_message = data.get('message', '')
    selected_bot_ids = data.get('selected_bots', [])
    web_search_enabled = data.get('web_search', False)
    individual_web_search = data.get('individual_web_search', False)
    user_system_prompt = data.get('system_prompt', '')
    session_id = request.sid  # Get unique session ID
    
    # Reset stop flag for this session
    stop_flags[session_id] = False
    
    # Initialize conversation history for this session if not exists
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    # Add user message to conversation history
    conversation_histories[session_id].append({
        "role": "user",
        "content": user_message
    })
    
    # Build chat history for API (exclude the current user message, it will be sent separately)
    chat_history = []
    for msg in conversation_histories[session_id][:-1]:  # Exclude last message (current one)
        chat_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Log user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    search_status = " (with web search)" if web_search_enabled else ""
    if individual_web_search:
        search_status += " (individual search enabled)"
    socketio.emit('console_log', {
        'message': f"[{timestamp}] User message received{search_status}, distributing to {len(selected_bot_ids)} AI(s)"
    })
    
    # Perform web search once if enabled
    web_search_results = None
    if web_search_enabled:
        socketio.emit('console_log', {
            'message': f"[{timestamp}] Performing web search..."
        })
        web_search_results = perform_web_search_once(user_message)
        
        # Display web search results in a collapsible format
        converted_search_results = convert_to_traditional_chinese(web_search_results)
        socketio.emit('web_search_results', {
            'results': converted_search_results,
            'timestamp': timestamp
        })
        
        socketio.emit('console_log', {
            'message': f"[{timestamp}] Web search completed"
        })
    
    # Get bot configurations
    bots = load_bots()
    
    # Create a bot lookup dictionary for faster access
    bots_dict = {bot['id']: bot for bot in bots}
    
    # Process each selected bot sequentially in the order specified by selected_bot_ids
    current_round_responses = []
    for bot_id in selected_bot_ids:
        # Check stop flag
        if stop_flags.get(session_id, False):
            socketio.emit('console_log', {
                'message': f"[{datetime.now().strftime('%H:%M:%S')}] Generation stopped by user"
            })
            break
        
        if bot_id in bots_dict:
            bot = bots_dict[bot_id]
            # Log that AI is processing
            socketio.emit('console_log', {
                'message': f"[{timestamp}] {bot['name']} started processing..."
            })
            
            # Emit 'running' status for this AI
            socketio.emit('ai_status', {
                'bot_id': bot['id'],
                'status': 'running'
            })
            
            # Check if this bot supports images
            support_images = bot.get('support_images', False)
            
            # Process this bot synchronously, passing previous responses in this round and web search results
            response = simulate_ai_response(
                bot['name'], 
                bot['id'], 
                user_message, 
                selected_bot_ids, 
                session_id, 
                chat_history,
                current_round_responses,
                web_search_results,
                support_images,
                individual_web_search,
                user_system_prompt
            )
            
            # Add this response to current round responses for next agent to see
            current_round_responses.append({
                'bot_name': bot['name'],
                'bot_id': bot['id'],
                'message': response
            })
    
    # Signal that all processing is done
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
