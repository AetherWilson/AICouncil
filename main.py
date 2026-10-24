from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_socketio import SocketIO, emit, join_room
import json
import time
import threading
import queue
import uuid
import importlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from datetime import datetime
import re
from GPT_handle import completion_response_stream, client
import os
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import base64
import logging
from services.config_store import ConfigStore, get_model_info as resolve_model_info
from services import memory_manager
from services.council_workflow import run_council, CouncilStopped, parse_object

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHAT_HISTORY_FOLDER'] = 'chat_history'
app.config['TEMP_CHAT_HISTORY_FOLDER'] = 'temp_chat_history'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")
logger = logging.getLogger(__name__)

config_store = ConfigStore(base_dir='.')
UPTEST_TIMEOUT_SECONDS = 30
SUPPORTED_UPLOAD_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'docx', 'docm', 'dotx', 'dotm'}
WORD_UPLOAD_EXTENSIONS = {'docx', 'docm', 'dotx', 'dotm'}

# Store all conversation state keyed by conversation_id
# {
#   conversation_id: {
#     messages: list,
#     is_generating: bool,
#     pending_message_id: str | None,
#     abort_event: threading.Event,
#     uploaded_documents: dict[str, dict]
#   }
# }
conversations = {}
CURRENT_CHAT_SCHEMA_VERSION = 4

# Prevent concurrent writes from overlapping background tasks.
persistence_lock = threading.Lock()
CHAT_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')


def _new_conversation_state():
    return {
        'messages': [],
        'is_generating': False,
        'pending_message_id': None,
        'abort_event': threading.Event(),
        'uploaded_documents': {},
        'run_group_counter': 0,
        'current_run_group_id': None,
        'redirect_messages': [],
        'run_lock': threading.Lock()
    }


def _log_internal_error(context, exc):
    logger.exception("%s: %s", context, exc)


def _is_valid_chat_id(chat_id):
    return bool(chat_id) and bool(CHAT_ID_PATTERN.fullmatch(chat_id))


def _resolve_chat_file_path(base_folder, chat_id):
    normalized_chat_id = str(chat_id or '').strip()
    if not _is_valid_chat_id(normalized_chat_id):
        raise ValueError('Invalid chat ID')

    base_path = os.path.abspath(base_folder)
    resolved_path = os.path.abspath(os.path.join(base_path, f'{normalized_chat_id}.json'))
    if os.path.commonpath([base_path, resolved_path]) != base_path:
        raise ValueError('Invalid chat ID path')
    return normalized_chat_id, resolved_path


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


def _extract_raw_markdown_from_ui_message(message):
    if not isinstance(message, dict):
        return str(message or '')

    for key in ('raw_markdown', 'content'):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ''


def _normalize_ui_message_for_storage(message):
    if not isinstance(message, dict):
        raw = str(message or '')
        return {
            'type': 'ai',
            'content': raw,
            'raw_markdown': raw
        }

    normalized = dict(message)
    raw_markdown = _extract_raw_markdown_from_ui_message(normalized)
    normalized['raw_markdown'] = raw_markdown
    return normalized


def _normalize_messages_for_storage(messages):
    return [_normalize_ui_message_for_storage(msg) for msg in (messages or [])]


def _replace_conversation_messages_from_ui(conversation_id, messages):
    """Replace in-memory conversation messages using UI message payload format."""
    conv = get_conversation(conversation_id)
    normalized_messages = messages if isinstance(messages, list) else []

    conv['messages'] = []
    for msg in normalized_messages:
        msg_type = str(msg.get('type') or '').strip().lower() if isinstance(msg, dict) else ''
        if msg_type == 'user':
            raw_user = _extract_raw_markdown_from_ui_message(msg)
            append_conversation_message(
                conversation_id,
                role='user',
                content=raw_user,
                raw_markdown=raw_user,
                id=(msg.get('id') if isinstance(msg, dict) else None) or uuid.uuid4().hex,
                run_group_id=(msg.get('run_group_id') if isinstance(msg, dict) else None)
            )
        elif msg_type == 'ai':
            bot_name = (msg.get('botName') if isinstance(msg, dict) else None) or (msg.get('sender') if isinstance(msg, dict) else None) or 'AI'
            bot_id = (msg.get('botId') if isinstance(msg, dict) else None) or 'unknown'
            raw_ai = _extract_raw_markdown_from_ui_message(msg)
            append_conversation_message(
                conversation_id,
                role='assistant',
                content=f"[{bot_name} ({bot_id})] {raw_ai}",
                raw_markdown=raw_ai,
                id=(msg.get('id') if isinstance(msg, dict) else None) or uuid.uuid4().hex,
                bot_name=bot_name,
                bot_id=bot_id,
                run_group_id=(msg.get('run_group_id') if isinstance(msg, dict) else None),
                run_id=(msg.get('run_id') if isinstance(msg, dict) else None) or (msg.get('id') if isinstance(msg, dict) else None),
                model_id=(msg.get('model_id') if isinstance(msg, dict) else None),
                role_name=(msg.get('role_name') if isinstance(msg, dict) else None),
                thinking=(msg.get('thinking') if isinstance(msg, dict) else '') or '',
                stream_status=(msg.get('stream_status') if isinstance(msg, dict) else '') or '',
                is_final_response=bool(msg.get('is_final_response', False)) if isinstance(msg, dict) else False,
                target_role=(msg.get('target_role') if isinstance(msg, dict) else None),
                debate_cycle=(msg.get('debate_cycle') if isinstance(msg, dict) else None),
                event_kind=(msg.get('event_kind') if isinstance(msg, dict) else None),
                council_phase=msg.get('council_phase'),
                council_round=msg.get('council_round'),
                council_payload=msg.get('council_payload'),
                is_subrole_hidden=bool(msg.get('is_subrole_hidden', False)) if isinstance(msg, dict) else False
            )

    _sync_run_group_counter(conversation_id, normalized_messages)
    return len(conv['messages'])


def _get_last_preview(messages):
    if not messages:
        return ''
    return (_extract_raw_markdown_from_ui_message(messages[-1]) or '')[:120]


def _is_final_response_ui_message(message):
    """Identify final-response UI messages."""
    if not isinstance(message, dict):
        return False

    if bool(message.get('is_final_response', False)):
        return True

    sender = str(
        message.get('sender')
        or message.get('bot_name')
        or message.get('botName')
        or ''
    ).lower()
    if 'leader' in sender and 'final response' in sender:
        return True

    role_name = str(message.get('role_name') or '').lower()
    return 'final response' in role_name


def _count_primary_chat_messages(messages):
    """Count only user messages and final-response AI messages for chat list badges."""
    count = 0
    for message in (messages or []):
        if not isinstance(message, dict):
            continue

        msg_type = str(message.get('type') or '').lower()
        if msg_type == 'user':
            count += 1
            continue

        if msg_type == 'ai' and _is_final_response_ui_message(message):
            count += 1

    return count


def migrate_chat_payload(chat_data):
    if not isinstance(chat_data, dict):
        return False

    changed = False
    if chat_data.get('schema_version') != CURRENT_CHAT_SCHEMA_VERSION:
        chat_data['schema_version'] = CURRENT_CHAT_SCHEMA_VERSION
        changed = True

    messages = chat_data.get('messages', [])
    if isinstance(messages, list):
        normalized = _normalize_messages_for_storage(messages)
        if normalized != messages:
            chat_data['messages'] = normalized
            changed = True

    return changed


def _migrate_chat_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        if migrate_chat_payload(chat_data):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            return True
    except Exception as e:
        print(f"Chat migration error for {filepath}: {e}")
    return False


def migrate_chat_files_once():
    updated = 0
    scanned = 0
    for folder in (app.config['CHAT_HISTORY_FOLDER'], app.config['TEMP_CHAT_HISTORY_FOLDER']):
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if not filename.endswith('.json'):
                continue
            scanned += 1
            if _migrate_chat_file(os.path.join(folder, filename)):
                updated += 1
    if scanned:
        print(f"Chat schema migration checked {scanned} file(s), updated {updated}.")


def _extract_run_group_sequence(run_group_id):
    if not run_group_id:
        return None
    match = re.search(r'-(\d{6})$', str(run_group_id))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _sync_run_group_counter(conversation_id, messages):
    conv = get_conversation(conversation_id)
    if not conv:
        return 0

    max_seq = int(conv.get('run_group_counter', 0) or 0)
    for msg in (messages or []):
        if not isinstance(msg, dict):
            continue
        seq = _extract_run_group_sequence(msg.get('run_group_id'))
        if seq is not None and seq > max_seq:
            max_seq = seq
    conv['run_group_counter'] = max_seq
    return max_seq


def _build_run_group_id(conversation_id):
    conv = get_conversation(conversation_id)
    conv['run_group_counter'] = int(conv.get('run_group_counter', 0) or 0) + 1
    counter = conv['run_group_counter']
    chat_token = re.sub(r'[^a-zA-Z0-9]+', '', str(conversation_id or ''))[:8] or 'chat'
    return f"rungrp-{int(time.time() * 1000)}-{chat_token}-{counter:06d}"


# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Create chat history directory if it doesn't exist
os.makedirs(app.config['CHAT_HISTORY_FOLDER'], exist_ok=True)
# Create temp chat history directory if it doesn't exist
os.makedirs(app.config['TEMP_CHAT_HISTORY_FOLDER'], exist_ok=True)

# One-time migration to keep persisted chat messages on the current schema.
migrate_chat_files_once()

# Load council role configurations
def load_config():
    return config_store.load_config()

def load_models():
    """Load available models from model.json"""
    return config_store.load_models()

def get_model_info(model_id):
    """Get model details by ID from model.json"""
    return resolve_model_info(load_models(), model_id)


def resolve_uptest_model_input(raw_model_name):
    """Resolve user-provided model text to a model id and display name.

    Accepts either exact model id or model display name (case-insensitive).
    Falls back to raw text as model id when no match is found.
    """
    text = str(raw_model_name or '').strip()
    if not text:
        return '', ''

    models = load_models()
    lowered = text.lower()

    for model in models:
        model_id = str(model.get('id', '') or '').strip()
        if model_id == text:
            return model_id, str(model.get('name', model_id) or model_id)

    for model in models:
        model_name = str(model.get('name', '') or '').strip()
        if model_name and model_name.lower() == lowered:
            model_id = str(model.get('id', '') or '').strip() or text
            return model_id, model_name

    return text, text

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


def _extract_document_payload(filepath, filename, pages_hint=None):
    """Extract normalized metadata/content for a stored document file."""
    file_size = os.path.getsize(filepath)
    file_ext = filename.lower().split('.')[-1]

    if file_ext == 'pdf':
        return {
            'filename': filename, 'content': '[PDF attached directly]',
            'filepath': filepath, 'pages': pages_hint, 'size': file_size, 'type': 'pdf',
        }

    if file_ext in WORD_UPLOAD_EXTENSIONS:
        try:
            docx_module = importlib.import_module('docx')
            Document = getattr(docx_module, 'Document', None)
            if Document is None:
                raise ValueError('Word extraction dependency is missing')
        except ImportError as exc:
            raise ValueError('Word extraction dependency is missing') from exc

        try:
            word_doc = Document(filepath)
        except Exception as exc:
            raise ValueError(f'Failed to parse Word file: {filename}') from exc

        text_chunks = []
        for para in word_doc.paragraphs:
            para_text = (para.text or '').strip()
            if para_text:
                text_chunks.append(para_text)

        for table in word_doc.tables:
            for row in table.rows:
                cells = [
                    (cell.text or '').strip()
                    for cell in row.cells
                    if (cell.text or '').strip()
                ]
                if cells:
                    text_chunks.append(' | '.join(cells))

        extracted_text = '\n'.join(text_chunks).strip()
        if not extracted_text:
            extracted_text = '[No extractable text found in Word document.]'

        return {
            'filename': filename,
            'content': '[Word document uploaded]',
            'filepath': filepath,
            'size': file_size,
            'type': 'word',
            'text': extracted_text
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
        'height': doc_payload.get('height'),
    }


def _coerce_message_attachments(raw_attachments):
    normalized = []
    if not isinstance(raw_attachments, list):
        return normalized

    for item in raw_attachments[:30]:
        if not isinstance(item, dict):
            continue
        filename = str(item.get('filename') or '').strip()
        if not filename:
            continue
        normalized.append({
            'filename': filename,
            'type': str(item.get('type') or '').strip().lower(),
            'pages': item.get('pages'),
        })
    return normalized


def _rehydrate_uploaded_documents_from_attachments(conv, attachments):
    if not isinstance(conv, dict):
        return 0

    uploaded_docs = conv.get('uploaded_documents')
    if not isinstance(uploaded_docs, dict):
        conv['uploaded_documents'] = {}
        uploaded_docs = conv['uploaded_documents']

    restored_count = 0
    for doc_meta in attachments:
        filename = str(doc_meta.get('filename') or '').strip()
        if not filename or filename in uploaded_docs:
            continue

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.isfile(filepath):
            continue

        try:
            doc_payload = _extract_document_payload(
                filepath,
                filename,
                pages_hint=doc_meta.get('pages')
            )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            _log_internal_error(f'rehydrate_documents skipped {filename}', exc)
            continue

        uploaded_docs[filename] = doc_payload
        restored_count += 1

    return restored_count

@app.route('/')
def index():
    # Clean up old temporary chats on page load
    cleanup_old_temp_chats()
    return render_template('index.html')

@app.route('/api/council', methods=['GET'])
def get_council():
    """Return the configured seats in speaking order, without specialist roles."""
    try:
        settings = load_config()['council']
        return jsonify({member['id']: {
            'model_id': member['model'], 'model_name': get_model_info(member['model']).get('name', member['model']),
            'name': member['name'], 'is_chair': member['id'] == settings['chair'], 'order': index + 1,
            'support_images': True, 'support_pdf_input': True,
        } for index, member in enumerate(settings['members'])})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


def _probe_model_latency(model_id, timeout_seconds):
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {'role': 'system', 'content': 'You are a health check. Reply very briefly.'},
                {'role': 'user', 'content': 'hello'}
            ],
            temperature=0,
            max_tokens=10,
            timeout=timeout_seconds
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        choices = getattr(response, 'choices', None) or []
        content = ''
        if choices:
            content = str((choices[0].message.content or '')).strip()

        if not content:
            return {
                'status': 'timeout',
                'elapsed_ms': None,
                'reply': '',
                'error': 'Empty response'
            }

        return {
            'status': 'ok',
            'elapsed_ms': elapsed_ms,
            'reply': content,
            'error': ''
        }
    except Exception as exc:
        return {
            'status': 'timeout',
            'elapsed_ms': None,
            'reply': '',
            'error': str(exc)
        }


@app.route('/api/backend/uptest', methods=['POST'])
def backend_uptest():
    payload = request.get_json(silent=True) or {}
    requested_timeout = payload.get('timeout_seconds', UPTEST_TIMEOUT_SECONDS)
    requested_model = payload.get('model_name', '')
    try:
        timeout_seconds = int(requested_timeout)
    except (TypeError, ValueError):
        timeout_seconds = UPTEST_TIMEOUT_SECONDS
    timeout_seconds = max(5, min(timeout_seconds, 120))

    role_rows = []
    selected_model_id, selected_model_name = resolve_uptest_model_input(requested_model)

    if selected_model_id:
        role_rows.append({
            'role': '__custom_model__',
            'role_label': 'Custom Model',
            'model_id': selected_model_id,
            'model_name': selected_model_name or selected_model_id,
            'status': 'timeout',
            'elapsed_ms': None,
            'reply': '',
            'error': ''
        })
    else:
        config = load_config()
        for member in config['council']['members']:
            role_name = member['id']
            model_id = member['model']
            model_info = get_model_info(model_id) if model_id else {}
            role_rows.append({
                'role': role_name,
                'role_label': member['name'],
                'model_id': model_id,
                'model_name': model_info.get('name', model_id) if model_id else 'Not configured',
                'status': 'timeout',
                'elapsed_ms': None,
                'reply': '',
                'error': ''
            })

    def generate_rows():
        futures = {}
        completed_roles = set()

        with ThreadPoolExecutor(max_workers=max(1, min(6, len(role_rows)))) as pool:
            for row in role_rows:
                if not row['model_id']:
                    row['status'] = 'timeout'
                    row['error'] = 'Model not configured'
                    completed_roles.add(row['role'])
                    yield json.dumps({'type': 'result', 'result': row}, ensure_ascii=False) + '\n'
                    continue
                future = pool.submit(_probe_model_latency, row['model_id'], timeout_seconds)
                futures[future] = row

            try:
                for future in as_completed(list(futures.keys()), timeout=timeout_seconds + 5):
                    row = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            'status': 'timeout',
                            'elapsed_ms': None,
                            'reply': '',
                            'error': str(exc)
                        }
                    row.update(result)
                    completed_roles.add(row['role'])
                    yield json.dumps({'type': 'result', 'result': row}, ensure_ascii=False) + '\n'
            except FutureTimeoutError:
                pass

            for row in role_rows:
                if row['role'] in completed_roles:
                    continue
                row.update({
                    'status': 'timeout',
                    'elapsed_ms': None,
                    'reply': '',
                    'error': 'Timeout'
                })
                yield json.dumps({'type': 'result', 'result': row}, ensure_ascii=False) + '\n'

        ok_count = sum(1 for row in role_rows if row.get('status') == 'ok')
        timeout_count = len(role_rows) - ok_count
        yield json.dumps({
            'type': 'done',
            'timeout_seconds': timeout_seconds,
            'model_name': selected_model_id,
            'ok_count': ok_count,
            'timeout_count': timeout_count,
            'total': len(role_rows)
        }, ensure_ascii=False) + '\n'

    return Response(
        stream_with_context(generate_rows()),
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all available models from model.json"""
    models = load_models()
    enabled = [m for m in models if m.get('enabled', True)]
    return jsonify(enabled)

@app.route('/api/upload_document', methods=['POST'])
def upload_document():
    """Handle PDF, image, and Word OpenXML upload and store normalized metadata"""
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
    if not filename:
        return jsonify({'success': False, 'error': 'Invalid file name'})

    file_ext = filename.lower().split('.')[-1]
    if file_ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        return jsonify({'success': False, 'error': 'Only PDF, JPG, JPEG, PNG, DOCX, DOCM, DOTX, and DOTM files are allowed'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        doc_payload = _extract_document_payload(filepath, filename)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        _log_internal_error('upload_document failed', exc)
        return jsonify({'success': False, 'error': 'Failed to process uploaded file'})
    except Exception as exc:
        _log_internal_error('upload_document unexpected failure', exc)
        return jsonify({'success': False, 'error': 'Failed to process uploaded file'})

    conv['uploaded_documents'][filename] = doc_payload
    return jsonify({
        'success': True,
        'document': _build_document_response(doc_payload)
    })

@app.route('/api/restore_documents', methods=['POST'])
def restore_documents():
    """Re-register previously uploaded documents from a saved chat"""
    data = request.get_json(silent=True) or {}
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

        try:
            doc_payload = _extract_document_payload(
                filepath,
                filename,
                pages_hint=doc_meta.get('pages')
            )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            _log_internal_error(f'restore_documents skipped {filename}', exc)
            continue

        conv['uploaded_documents'][filename] = doc_payload
        restored.append(_build_document_response(doc_payload))

    return jsonify({'success': True, 'documents': restored})

@app.route('/api/remove_document', methods=['POST'])
def remove_document():
    """Remove uploaded document"""
    data = request.get_json(silent=True) or {}
    conversation_id = data.get('chat_id')
    if not conversation_id:
        return jsonify({'success': False, 'error': 'chat_id is required'})
    conv = get_conversation(conversation_id)
    filename = data.get('filename')

    if filename in conv['uploaded_documents']:
        del conv['uploaded_documents'][filename]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as exc:
                _log_internal_error(f'remove_document failed to delete {filename}', exc)
                return jsonify({'success': False, 'error': 'Failed to remove file'})
        return jsonify({'success': True})

    return jsonify({'success': False, 'error': 'File not found'})

@app.route('/api/save_chat', methods=['POST'])
def save_chat():
    """Save a chat session to disk"""
    data = request.get_json(silent=True) or {}
    try:
        chat_id, filepath = _resolve_chat_file_path(app.config['CHAT_HISTORY_FOLDER'], data.get('id'))
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid chat ID'})

    chat_data = {
        'id': chat_id,
        'name': data.get('name'),
        'schema_version': CURRENT_CHAT_SCHEMA_VERSION,
        'messages': _normalize_messages_for_storage(data.get('messages', [])),
        'selectedBots': data.get('selectedBots', []),
        'systemPrompt': data.get('systemPrompt', ''),
        'timestamp': data.get('timestamp', datetime.now().isoformat()),
        'uploadedDocuments': data.get('uploadedDocuments', [])
    }

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        _log_internal_error('save_chat failed', exc)
        return jsonify({'success': False, 'error': 'Failed to save chat'})

    return jsonify({'success': True, 'id': chat_id})

@app.route('/api/list_chats', methods=['GET'])
def list_chats():
    """List all saved chat sessions"""
    try:
        chat_files = [f for f in os.listdir(app.config['CHAT_HISTORY_FOLDER']) if f.endswith('.json')]
    except OSError as exc:
        _log_internal_error('list_chats failed to list directory', exc)
        return jsonify({'success': False, 'error': 'Failed to load chat list', 'chats': []})

    chats = []
    for filename in chat_files:
        filepath = os.path.join(app.config['CHAT_HISTORY_FOLDER'], filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as exc:
            _log_internal_error(f'list_chats invalid JSON in {filename}', exc)
            continue
        except OSError as exc:
            _log_internal_error(f'list_chats cannot read {filename}', exc)
            continue

        chats.append({
            'id': chat_data.get('id'),
            'name': chat_data.get('name'),
            'timestamp': chat_data.get('timestamp'),
            'messageCount': _count_primary_chat_messages(chat_data.get('messages', [])),
            'lastPreview': _get_last_preview(chat_data.get('messages', [])),
            'isGenerating': get_conversation(chat_data.get('id')).get('is_generating', False)
        })

    chats.sort(key=lambda x: x.get('timestamp') or '', reverse=True)
    return jsonify({'success': True, 'chats': chats})

@app.route('/api/load_chat/<chat_id>', methods=['GET'])
def load_chat(chat_id):
    """Load a specific chat session"""
    try:
        normalized_chat_id, filepath = _resolve_chat_file_path(app.config['CHAT_HISTORY_FOLDER'], chat_id)
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid chat ID'})

    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'Chat not found'})

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
    except json.JSONDecodeError as exc:
        _log_internal_error(f'load_chat invalid JSON for {normalized_chat_id}', exc)
        return jsonify({'success': False, 'error': 'Failed to load chat'})
    except OSError as exc:
        _log_internal_error(f'load_chat read failed for {normalized_chat_id}', exc)
        return jsonify({'success': False, 'error': 'Failed to load chat'})

    if migrate_chat_payload(chat_data):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            _log_internal_error(f'load_chat migration write failed for {normalized_chat_id}', exc)

    # Keep persisted message payload as source of truth for UI rendering.
    # Frontend-only thinking timeline entries are stored in chat JSON and would be
    # lost if we swapped to backend in-memory history here.

    target_conv = get_conversation(normalized_chat_id)
    if not target_conv.get('is_generating'):
        hydrated_count = _replace_conversation_messages_from_ui(
            normalized_chat_id,
            chat_data.get('messages', [])
        )
        emit_log_ts = datetime.now().strftime("%H:%M:%S")
        logger.info("[%s] Hydrated conversation %s from load_chat (%s messages)", emit_log_ts, normalized_chat_id, hydrated_count)

    _sync_run_group_counter(normalized_chat_id, chat_data.get('messages', []))
    chat_data['isGenerating'] = get_conversation(normalized_chat_id).get('is_generating', False)
    return jsonify({'success': True, 'chat': chat_data})

@app.route('/api/delete_chat', methods=['POST'])
def delete_chat():
    """Delete a saved chat session"""
    data = request.get_json(silent=True) or {}
    try:
        _, filepath = _resolve_chat_file_path(app.config['CHAT_HISTORY_FOLDER'], data.get('id'))
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid chat ID'})

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError as exc:
            _log_internal_error('delete_chat failed', exc)
            return jsonify({'success': False, 'error': 'Failed to delete chat'})
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Chat not found'})

@app.route('/api/save_temp_chat', methods=['POST'])
def save_temp_chat():
    """Save a temporary chat session that will be auto-deleted next day"""
    data = request.get_json(silent=True) or {}
    requested_chat_id = data.get('id') or ('temp_' + str(int(time.time() * 1000)))
    try:
        chat_id, filepath = _resolve_chat_file_path(app.config['TEMP_CHAT_HISTORY_FOLDER'], requested_chat_id)
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid chat ID'})

    chat_data = {
        'id': chat_id,
        'name': 'Temporary Chat',
        'schema_version': CURRENT_CHAT_SCHEMA_VERSION,
        'messages': _normalize_messages_for_storage(data.get('messages', [])),
        'selectedBots': data.get('selectedBots', []),
        'systemPrompt': data.get('systemPrompt', ''),
        'timestamp': datetime.now().isoformat(),
        'isTemporary': True,
        'uploadedDocuments': data.get('uploadedDocuments', [])
    }

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        _log_internal_error('save_temp_chat failed', exc)
        return jsonify({'success': False, 'error': 'Failed to save chat'})

    return jsonify({'success': True, 'id': chat_id})

@app.route('/api/list_temp_chats', methods=['GET'])
def list_temp_chats():
    """List all temporary chat sessions from today"""
    temp_folder = app.config['TEMP_CHAT_HISTORY_FOLDER']
    try:
        chat_files = [f for f in os.listdir(temp_folder) if f.endswith('.json')]
    except OSError as exc:
        _log_internal_error('list_temp_chats failed to list directory', exc)
        return jsonify({'success': False, 'error': 'Failed to load chat list', 'chats': []})

    chats = []
    for filename in chat_files:
        filepath = os.path.join(temp_folder, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as exc:
            _log_internal_error(f'list_temp_chats invalid JSON in {filename}', exc)
            continue
        except OSError as exc:
            _log_internal_error(f'list_temp_chats cannot read {filename}', exc)
            continue

        chats.append({
            'id': chat_data.get('id'),
            'name': 'Temp Chat',
            'timestamp': chat_data.get('timestamp'),
            'messageCount': _count_primary_chat_messages(chat_data.get('messages', [])),
            'isTemporary': True,
            'lastPreview': _get_last_preview(chat_data.get('messages', [])),
            'isGenerating': get_conversation(chat_data.get('id')).get('is_generating', False)
        })

    chats.sort(key=lambda x: x.get('timestamp') or '', reverse=True)
    return jsonify({'success': True, 'chats': chats})

@app.route('/api/load_temp_chat/<chat_id>', methods=['GET'])
def load_temp_chat(chat_id):
    """Load a specific temporary chat session"""
    try:
        normalized_chat_id, filepath = _resolve_chat_file_path(app.config['TEMP_CHAT_HISTORY_FOLDER'], chat_id)
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid chat ID'})

    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'Temporary chat not found'})

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
    except json.JSONDecodeError as exc:
        _log_internal_error(f'load_temp_chat invalid JSON for {normalized_chat_id}', exc)
        return jsonify({'success': False, 'error': 'Failed to load chat'})
    except OSError as exc:
        _log_internal_error(f'load_temp_chat read failed for {normalized_chat_id}', exc)
        return jsonify({'success': False, 'error': 'Failed to load chat'})

    if migrate_chat_payload(chat_data):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            _log_internal_error(f'load_temp_chat migration write failed for {normalized_chat_id}', exc)

    # Keep persisted message payload as source of truth for UI rendering.
    # Frontend-only thinking timeline entries are stored in chat JSON and would be
    # lost if we swapped to backend in-memory history here.

    target_conv = get_conversation(normalized_chat_id)
    if not target_conv.get('is_generating'):
        hydrated_count = _replace_conversation_messages_from_ui(
            normalized_chat_id,
            chat_data.get('messages', [])
        )
        emit_log_ts = datetime.now().strftime("%H:%M:%S")
        logger.info("[%s] Hydrated conversation %s from load_temp_chat (%s messages)", emit_log_ts, normalized_chat_id, hydrated_count)

    _sync_run_group_counter(normalized_chat_id, chat_data.get('messages', []))
    chat_data['isGenerating'] = get_conversation(normalized_chat_id).get('is_generating', False)
    return jsonify({'success': True, 'chat': chat_data})

@app.route('/api/delete_temp_chat', methods=['POST'])
def delete_temp_chat():
    """Delete a temporary chat session"""
    data = request.get_json(silent=True) or {}
    try:
        _, filepath = _resolve_chat_file_path(app.config['TEMP_CHAT_HISTORY_FOLDER'], data.get('id'))
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid chat ID'})

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError as exc:
            _log_internal_error('delete_temp_chat failed', exc)
            return jsonify({'success': False, 'error': 'Failed to delete chat'})
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Temporary chat not found'})

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
    except OSError as exc:
        _log_internal_error(f'encode_image_to_base64 failed for {filepath}', exc)
        return None

def _normalize_history_context_mode(value):
    normalized = str(value or '').strip().lower()
    if normalized in {'all', 'final_only'}:
        return normalized
    return 'final_only'


def _is_final_response_history_message(message):
    if not isinstance(message, dict):
        return False

    if str(message.get('role') or '').lower() != 'assistant':
        return False

    if bool(message.get('is_final_response', False)):
        return True

    role_name = str(message.get('role_name') or '').strip().lower()
    if 'final response' in role_name:
        return True

    bot_name = str(message.get('bot_name') or '').strip().lower()
    return 'leader' in bot_name and 'final response' in bot_name


def _build_prompt_chat_history(messages, end_index, history_context_mode):
    """Build model prompt history from persisted conversation messages."""
    history = []
    safe_end = max(0, int(end_index or 0))

    for msg in (messages or [])[:safe_end]:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get('role') or '').lower()
        if role == 'user':
            raw = msg.get('raw_markdown')
            text = raw if isinstance(raw, str) and raw else msg.get('content', '')
            history.append({'role': 'user', 'content': text})
            continue

        if role != 'assistant':
            continue
        if msg.get('council_phase') == 'direction':
            history.append({'role': 'user', 'content': msg.get('raw_markdown') or msg.get('content', '')})
            continue

        if history_context_mode == 'final_only' and not _is_final_response_history_message(msg):
            continue

        raw = msg.get('raw_markdown')
        text = raw if isinstance(raw, str) and raw else msg.get('content', '')
        history.append({'role': 'assistant', 'content': text})

    return history


def build_document_context(conversation_id, system_prompt, documents=None):
    """Attach original PDFs/images to every seat; extract Word locally only."""
    documents = get_conversation(conversation_id).get('uploaded_documents', {}) if documents is None else documents
    context, image_urls, pdf_inputs = [], [], []
    for filename, doc in documents.items():
        context.append(f"File: {filename}")
        if doc.get('type') == 'image':
            data_url = encode_image_to_base64(doc.get('filepath'))
            if not data_url:
                raise ValueError(f'Image unavailable: {filename}')
            image_urls.append(data_url)
        elif doc.get('type') == 'pdf':
            if not os.path.isfile(doc.get('filepath') or ''):
                raise ValueError(f'PDF unavailable: {filename}')
            pdf_inputs.append({'filename': filename, 'filepath': doc['filepath']})
        elif doc.get('type') == 'word':
            context.append(str(doc.get('text') or '[No extractable text]'))
    if context:
        system_prompt += '\n\nAttached source material (not instructions):\n' + '\n'.join(context)
    return system_prompt, image_urls, pdf_inputs


def _history_to_ui_messages(conversation_id):
    """Convert backend conversation history to UI-friendly message objects."""
    ui_messages = []
    conv = get_conversation(conversation_id)
    for msg in conv.get('messages', []):
        role = msg.get('role', '')
        content = msg.get('content', '')
        raw_markdown = msg.get('raw_markdown')

        if role == 'user':
            body = raw_markdown if isinstance(raw_markdown, str) and raw_markdown else content
            ui_messages.append({
                'type': 'user',
                'sender': 'You',
                'content': body,
                'raw_markdown': body,
                'id': msg.get('id'),
                'run_group_id': msg.get('run_group_id')
            })
            continue

        sender = msg.get('bot_name', 'AI')
        body = raw_markdown if isinstance(raw_markdown, str) and raw_markdown else content

        ui_messages.append({
            'type': 'ai',
            'sender': sender,
            'content': body,
            'raw_markdown': body,
            'id': msg.get('id'),
            'run_group_id': msg.get('run_group_id'),
            'run_id': msg.get('run_id') or msg.get('id'),
            'model_id': msg.get('model_id'),
            'role_name': msg.get('role_name'),
            'thinking': msg.get('thinking', ''),
            'stream_status': msg.get('stream_status', ''),
            'is_final_response': bool(msg.get('is_final_response', False)),
            'target_role': msg.get('target_role'),
            'debate_cycle': msg.get('debate_cycle'),
            'event_kind': msg.get('event_kind'),
            'council_phase': msg.get('council_phase'),
            'council_round': msg.get('council_round'),
            'council_payload': msg.get('council_payload'),
            'is_subrole_hidden': bool(msg.get('is_subrole_hidden', False))
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
            chat_data['schema_version'] = CURRENT_CHAT_SCHEMA_VERSION
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
                    'height': doc.get('height'),
                }
                for doc in conv.get('uploaded_documents', {}).values()
            ]
            if not is_saved:
                chat_data['isTemporary'] = True

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Snapshot save error for {session_id}: {e}")


PHASE_LABELS = {
    'agenda': 'Agenda', 'final': 'Final Response', 'proposal': 'Independent proposal', 'discussion': 'Discussion', 'draft': 'Chair draft',
    'vote': 'Ballot', 'tally': 'Vote result', 'direction': 'User update', 'failure': 'Turn failed',
}


def _council_message_meta(member, phase, number, run_group_id, payload=None, final=False):
    name = member['name'] if member else 'Council'
    model = member['model'] if member else ''
    label = 'Final Response' if final else PHASE_LABELS.get(phase, phase)
    return {
        'bot_name': f"{name} · {label}{f' {number}' if number else ''}" + (f' ({model})' if model else ''),
        'bot_id': f'council-{uuid.uuid4().hex[:12]}',
        'message_id': uuid.uuid4().hex, 'run_id': uuid.uuid4().hex,
        'run_group_id': run_group_id, 'role_name': member['id'] if member else 'council',
        'model_id': model, 'is_final_response': final,
        'is_council_intermediate': not final, 'event_kind': f'council_{phase}',
        'council_phase': phase, 'council_round': number, 'council_payload': payload or {},
        'debate_cycle': number, 'timestamp': datetime.now().strftime('%H:%M:%S'),
    }


def _store_council_message(conversation_id, meta, text, status='done'):
    fields = {key: value for key, value in meta.items() if key not in {'message_id', 'timestamp'}}
    append_conversation_message(conversation_id, 'assistant', text,
                                raw_markdown=text, id=meta['message_id'], stream_status=status, **fields)


def _publish_council_message(conversation_id, emit_chat, member, phase, number, group, text, payload=None, final=False):
    meta = _council_message_meta(member, phase, number, group, payload, final)
    _store_council_message(conversation_id, meta, text)
    emit_chat('ai_response_start', meta)
    emit_chat('ai_response_end', dict(meta, message=text, thinking=''))


def _request_council_turn(*, conversation_id, emit_chat, member, system, prompt,
                          phase, number, group, history, documents, timeout, aborted):
    """One bounded turn. Only this controller writes messages or emits UI events.

    Timed-out worker output is discarded; it cannot append a late turn or vote.
    Votes use the same direct-file transport, but have no public message until reveal.
    """
    private = phase == 'vote'
    system, images, pdfs = build_document_context(conversation_id, system, documents)
    meta = _council_message_meta(member, phase, number, group)
    conv = get_conversation(conversation_id)
    events = queue.Queue()
    cancel = threading.Event()

    def worker():
        stream = completion_response_stream(
            model=member['model'], system_prompt=system, user_prompt=prompt,
            chat_history=history, temperature=0.3 if private else 0.7,
            image_urls=images or None, pdf_inputs=pdfs or None,
            timeout_seconds=timeout, cancel_event=cancel,
        )
        try:
            for kind, chunk in stream:
                if cancel.is_set():
                    break
                events.put(('chunk', kind, chunk))
            events.put(('done', None, None))
        except Exception as exc:
            events.put(('error', None, exc))
        finally:
            stream.close()

    emit_chat('council_status', {'role': member['id'], 'status': 'running',
                               'run_group_id': group, 'council_phase': phase, 'council_round': number})
    if not private:
        _store_council_message(conversation_id, meta, '', 'running')
        conv['pending_message_id'] = meta['message_id']
        emit_chat('ai_response_start', meta)
    deadline = time.monotonic() + timeout
    threading.Thread(target=worker, daemon=True).start()
    text = ''
    status = 'done'
    last_save = time.monotonic()
    try:
        while True:
            if aborted():
                raise CouncilStopped()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'{member["name"]} exceeded {timeout}s turn limit')
            try:
                event, kind, value = events.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if event == 'error':
                raise value
            if event == 'done':
                break
            # The council discusses submitted answers, never model reasoning traces.
            if kind != 'content':
                continue
            text += str(value)
            if not private:
                update_message_fields(conversation_id, meta['message_id'], content=text, raw_markdown=text)
                emit_chat('ai_response_chunk', dict(meta, chunk=str(value)))
                if time.monotonic() - last_save >= 1:
                    persist_chat_snapshot(conversation_id)
                    last_save = time.monotonic()
        if aborted():
            raise CouncilStopped()
        if not text.strip():
            raise RuntimeError('Model returned an empty answer')
        if phase == 'discussion':
            parsed = parse_object(text)
            if isinstance(parsed.get('contribution'), str) and parsed['contribution'].strip():
                display_text = parsed['contribution']
            else:
                display_text = text
        else:
            display_text = text
        return text
    except CouncilStopped:
        status = 'stopped'
        display_text = text + '\n\n[Stopped; incomplete turn.]'
        raise
    except Exception:
        status = 'error'
        display_text = text + '\n\n[Turn failed or timed out; incomplete output is excluded from deliberation.]'
        raise
    finally:
        cancel.set()
        if not private:
            update_message_fields(conversation_id, meta['message_id'], content=display_text,
                                  raw_markdown=display_text, stream_status=status)
            emit_chat('ai_response_end', dict(meta, message=display_text, thinking='',
                                            stream_status=status, stopped=status == 'stopped'))
            conv['pending_message_id'] = None
        emit_chat('council_status', {'role': member['id'], 'status': status,
                                   'run_group_id': group, 'council_phase': phase})
        persist_chat_snapshot(conversation_id)


@socketio.on('join_chat')
def handle_join_chat(data):
    if data.get('chat_id'):
        join_room(data['chat_id'])


@socketio.on('send_message')
def handle_message_wrapper(data):
    conversation_id = data.get('chat_id') or request.sid
    conv = get_conversation(conversation_id)
    with conv['run_lock']:
        if conv.get('is_generating'):
            if conv['abort_event'].is_set():
                emit('console_log', {'chat_id': conversation_id, 'message': 'Council is stopping; please wait for completion.'}, to=conversation_id)
                return
            text = str(data.get('message') or '').strip()
            if text:
                conv['redirect_messages'].append(text)
                emit('console_log', {'chat_id': conversation_id,
                                     'message': 'User update queued for the next council stage boundary.'}, to=conversation_id)
            return
        _rehydrate_uploaded_documents_from_attachments(conv, _coerce_message_attachments(data.get('attachments')))
        conv['abort_event'].clear()
        conv['redirect_messages'] = []
        conv['is_generating'] = True
    socketio.start_background_task(handle_message_task, data, conversation_id)


def handle_message_task(data, conversation_id):
    conv = get_conversation(conversation_id)
    group = _build_run_group_id(conversation_id)
    conv['current_run_group_id'] = group
    user_message = str(data.get('message') or '')
    user_system = str(data.get('system_prompt') or '')

    def emit_chat(event, payload=None):
        socketio.emit(event, dict(payload or {}, chat_id=conversation_id), to=conversation_id)

    def publish(phase, number, member, text, payload):
        if conv['abort_event'].is_set():
            raise CouncilStopped()
        _publish_council_message(conversation_id, emit_chat, member, phase, number, group, text, payload)
        persist_chat_snapshot(conversation_id, user_system)

    def take_redirects():
        with conv['run_lock']:
            updates = conv['redirect_messages'][:]
            conv['redirect_messages'].clear()
        return updates

    try:
        existing = str(data.get('existing_user_message_id') or '')
        index = next((i for i, msg in enumerate(conv['messages'])
                      if existing and msg.get('id') == existing and msg.get('role') == 'user'), None)
        if index is None:
            append_conversation_message(conversation_id, 'user', user_message,
                                        raw_markdown=user_message, run_group_id=group)
            index = len(conv['messages']) - 1
        else:
            msg = conv['messages'][index]
            user_message = msg.get('raw_markdown') or msg.get('content', '')
            msg['run_group_id'] = group
        config = load_config()
        settings = config['council']
        history = _build_prompt_chat_history(conv['messages'], index, _normalize_history_context_mode(config.get('history_context_mode', 'final_only')))
        documents = {name: dict(doc) for name, doc in conv.get('uploaded_documents', {}).items()}
        memory = memory_manager.build_memory_context(config)
        emit_chat('run_group_start', {'run_group_id': group, 'timestamp': datetime.now().strftime('%H:%M:%S')})
        publish('agenda', 0, None, '**Speaking order:** ' + ' → '.join(member['name'] for member in settings['members'])
                + f"\n\nUp to {settings['discussion_rounds']} discussion rounds; "
                + f"{settings['max_revisions']} draft revisions. Ballots are revealed together.", settings)

        def request_turn(member, system, prompt, phase, number, private):
            if user_system:
                system += '\n\nUser preferences:\n' + user_system
            if memory:
                system += '\n\n' + memory
            return _request_council_turn(
                conversation_id=conversation_id, emit_chat=emit_chat, member=member,
                system=system, prompt=prompt, phase=phase, number=number, group=group,
                history=history, documents=documents, timeout=settings['turn_timeout_seconds'],
                aborted=conv['abort_event'].is_set,
            )

        result = run_council(settings=settings, user_message=user_message, request_turn=request_turn,
                             publish=publish, stopped=conv['abort_event'].is_set, take_redirects=take_redirects)
        if conv['abort_event'].is_set():
            raise CouncilStopped()
        chair = next(member for member in settings['members'] if member['id'] == settings['chair'])
        _publish_council_message(conversation_id, emit_chat, chair, 'final', result.draft_number,
                                 group, result.markdown(), {'approved': result.approved, 'ballots': result.ballots,
                                                           'draft_number': result.draft_number}, final=True)
    except CouncilStopped:
        emit_chat('console_log', {'message': 'Council stopped. No further turns or final approval will be issued.'})
    except Exception as exc:
        _log_internal_error('Council workflow failed', exc)
        _publish_council_message(conversation_id, emit_chat, None, 'final', 0, group,
                                 f'Council could not complete: {exc}', {'approved': False}, final=True)
    finally:
        persist_chat_snapshot(conversation_id, user_system)
        conv['pending_message_id'] = None
        conv['current_run_group_id'] = None
        # Emit completion before accepting the next sitting, so stale completion
        # events cannot mark a newer sitting as finished.
        with conv['run_lock']:
            emit_chat('run_group_end', {'run_group_id': group, 'stopped': conv['abort_event'].is_set()})
            emit_chat('all_done', {'run_group_id': group})
            conv['is_generating'] = False


@socketio.on('stop_generation')
def handle_stop(data):
    """Handle stop generation request"""
    conversation_id = data.get('chat_id') if data and data.get('chat_id') else request.sid
    conv = get_conversation(conversation_id)
    conv['abort_event'].set()
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit('console_log', {
        'message': f"[{timestamp}] Stop requested - cancelling this council sitting",
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
    # Keep per-chat state alive after UI switches/disconnects.
    # Conversation lifecycle is tied to chat IDs and persisted history, not socket IDs.
    print('Client disconnected')

@socketio.on('clear_history')
def handle_clear_history(data=None):
    """Clear conversation history for the current session"""
    chat_id = data.get('chat_id') if data else None
    conversation_id = chat_id if chat_id else request.sid
    conv = get_conversation(conversation_id)
    if conv.get('is_generating'):
        conv['abort_event'].set()
        return
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

    loaded_count = _replace_conversation_messages_from_ui(conversation_id, messages)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    emit('console_log', {
        'message': f"[{timestamp}] Chat history loaded ({loaded_count} messages)",
        'chat_id': conversation_id
    }, to=conversation_id)

# ?�?�?�?Memory Management API Endpoints ?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

@app.route('/api/memories', methods=['GET'])
def get_memories():
    """Get all cross-chat memory entries"""
    try:
        config = load_config()
        entries = memory_manager.read_flat(config)
        enabled = memory_manager.is_enabled(config)
        auto_extract = memory_manager.auto_extract_enabled(config)
        return jsonify({
            'success': True,
            'memories': entries,
            'enabled': enabled,
            'auto_extract': auto_extract,
            'sections': memory_manager.SECTIONS
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/memories', methods=['POST'])
def add_memory_endpoint():
    """Add a new memory entry"""
    try:
        data = request.json
        section = data.get('section', 'Key Facts')
        content = data.get('content', '').strip()
        if not content:
            return jsonify({'success': False, 'error': 'Content is required'})

        config = load_config()
        entries = memory_manager.add_memory(section, content, config)
        return jsonify({'success': True, 'memories': entries})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/memories/<int:memory_id>', methods=['PUT'])
def update_memory_endpoint(memory_id):
    """Update an existing memory entry"""
    try:
        data = request.json
        content = data.get('content', '').strip()
        if not content:
            return jsonify({'success': False, 'error': 'Content is required'})

        config = load_config()
        entries = memory_manager.update_memory(memory_id, content, config)
        return jsonify({'success': True, 'memories': entries})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/memories/<int:memory_id>', methods=['DELETE'])
def delete_memory_endpoint(memory_id):
    """Delete a memory entry"""
    try:
        config = load_config()
        entries = memory_manager.delete_memory(memory_id, config)
        return jsonify({'success': True, 'memories': entries})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/memories/clear', methods=['POST'])
def clear_memories_endpoint():
    """Clear all memory entries"""
    try:
        config = load_config()
        memory_manager.clear_all(config)
        return jsonify({'success': True, 'memories': []})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("Starting Flask server...")
    print("Visit http://localhost:5000 in your browser")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
