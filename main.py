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
from GPT_handle import completion_response, completion_response_stream, convert_to_traditional_chinese, client
import os
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import base64
import logging
from services.config_store import ConfigStore, get_model_info as resolve_model_info
from services import memory_manager, skill_registry, skill_tool_runner

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHAT_HISTORY_FOLDER'] = 'chat_history'
app.config['TEMP_CHAT_HISTORY_FOLDER'] = 'temp_chat_history'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")
logger = logging.getLogger(__name__)

PDF_READER_SKILL_ID = 'pdfer-skill'
DEFAULT_PDF_READER_MODEL_ID = 'gemini-3.1-pro-preview'

COUNCIL_ROLES = ['Leader']
AGENT_SKILL_ROLE_NAMES = {
    'researcher-skill': 'ResearcherSkill',
    'creator-skill': 'CreatorSkill',
    'analyzer-skill': 'AnalyzerSkill',
    'verifier-skill': 'VerifierSkill',
    PDF_READER_SKILL_ID: 'PDFReaderSkill'
}
DEFAULT_SKILLS_CONFIG = {
    'enabled': True,
    'folder': 'skills',
    'max_files': 3,
    'max_chars_per_file': 2500,
    'max_total_chars': 7000
}
DEFAULT_MD_READER_CONFIG = {
    'enabled': True,
    'max_inventory_files': 40,
    'preview_lines_per_file': 20,
    'preview_chars_per_file': 1200
}
config_store = ConfigStore(base_dir='.')
UPTEST_TIMEOUT_SECONDS = 30
DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS = 30.0
LONG_INPUT_TOKEN_THRESHOLD = 10000
LONG_INPUT_FIRST_CHUNK_TIMEOUT_SECONDS = 60.0
APPROX_CHARS_PER_TOKEN = 4.0
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
CURRENT_CHAT_SCHEMA_VERSION = 3

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
        'agent_redirect_message': ''
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


def should_continue_streaming(conversation_id, pending_message_id):
    conv = get_conversation(conversation_id)
    return (
        conv is not None
        and conv.get('is_generating', False)
        and not conv.get('abort_event').is_set()
        and conv.get('pending_message_id') == pending_message_id
    )


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


def _build_fallback_role_label(role_label, fallback_model_id):
    if '(' in role_label and role_label.endswith(')'):
        return re.sub(r'\([^)]*\)$', f'({fallback_model_id})', role_label)
    return f"{role_label} ({fallback_model_id})"

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


def _resolve_pdf_reader_model_id(config, fallback_model_id=''):
    skills_cfg = config.get('skills', {}) if isinstance(config.get('skills', {}), dict) else {}
    model_map = skills_cfg.get('model_map', {}) if isinstance(skills_cfg.get('model_map', {}), dict) else {}
    mapped = str(model_map.get(PDF_READER_SKILL_ID) or '').strip()
    if mapped:
        return mapped

    fallback = str(fallback_model_id or '').strip()
    if fallback:
        return fallback

    return DEFAULT_PDF_READER_MODEL_ID


def _list_uploaded_pdf_filenames(conv):
    if not isinstance(conv, dict):
        return []
    documents = conv.get('uploaded_documents', {})
    if not isinstance(documents, dict):
        return []
    return [
        str(name)
        for name, doc in documents.items()
        if isinstance(doc, dict) and str(doc.get('type') or '').lower() == 'pdf'
    ]


def _resolve_lite_model_id(config, fallback_model_id=''):
    if isinstance(config, dict):
        lite_model_id = str(config.get('lite_model') or '').strip()
        if lite_model_id:
            return lite_model_id

    fallback = str(fallback_model_id or '').strip()
    if fallback:
        return fallback

    if isinstance(config, dict):
        return str(config.get('Leader') or '').strip()
    return ''


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

# Core agent prompts
LEADER_AGENT_ACTION_PROMPT = """You are a single AI agent that plans actions and can call skills.

Rules:
- Choose either a skill call or a final response in each turn.
- If you choose a skill call, return ONLY one JSON object using the skill_call schema. Skill calls, payloads, and intermediate agent steps are internal and never visible to the user.
- If you choose a final response, plain markdown text is allowed and preferred for quality (JSON wrapper is optional). Assume the user cannot see any skill calls or hidden workflow outputs.
- Use skills when factual grounding, analysis, or quality checks are needed.
- For pdfer-skill calls, always include args.filenames as a list of uploaded PDF filenames to read.
- For factual/explanatory/comparison/recommendation requests, default to researcher-skill.
- If there is even slight uncertainty about correctness, completeness, or freshness, call researcher-skill instead of final_response.
- "I can probably answer from memory" still counts as uncertainty for knowledge-dependent tasks.
- Use final_response only when the request is non-factual transformation (rewrite/format/translate) or after needed skill evidence is already gathered.
- When writing final_response, include all relevant facts, details, caveats, and conclusions from the called skills directly in the user-facing answer.
- Never imply the user can inspect skill/tool traces (avoid phrases like "as shown in the skill call" or "see above tool output").
- Prefer verifier-skill for logic/math content or when confidence is low.

Required JSON schema:
{
    "thought": "short planning thought",
    "requires_verifier": true,
    "confidence": 0.0,
    "action": {
        "type": "skill_call",
        "skill": "researcher-skill",
        "args": {
            "task": "what to do"
        },
        "reason": "why this skill is needed"
    }
}

Or for final answer:
{
    "thought": "short planning thought",
    "requires_verifier": false,
    "confidence": 0.0,
    "action": {
        "type": "final_response",
        "text": "final answer for the user in markdown"
    }
}

Alternative final-answer format (no JSON):
- Return the final user-facing answer directly in markdown.
"""

SKILL_EXECUTION_PROMPT_TEMPLATE = """You are executing one skill for an orchestrating Leader agent.

Skill id: {skill_id}
Skill description: {skill_description}

Skill instructions:
{skill_content}

Execution requirement:
- Complete the task using the skill guidance above.
- Return only one JSON object.
- Keep content concise and machine-usable.

JSON schema:
{{
    "result": "main output",
    "confidence": 0.0,
    "notes": ["optional notes"]
}}
"""

MARK_READER_PROMPT = """You are the MarkReader in an AI council. Your only job is to choose which markdown skill files are relevant for the Leader.

Rules:
- Use only file paths that appear in the provided inventory
- Choose up to max_files files, but pick fewer if relevance is weak
- Prefer files that directly help task routing or final synthesis quality
- Do not invent file paths
- Be concise

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{
    "selected_files": ["skills/example.md"],
    "reason": "Short reason for selection"
}"""


MEMORY_EXTRACT_PROMPT = """You are a memory management system. Your job is to analyze the conversation and manage the user's persistent cross-chat memory: adding new facts, updating outdated ones, and removing entries that are no longer relevant.

Rules for ADDING new memories:
- Extract ONLY facts that would be useful in future UNRELATED conversations.
- Focus on: 
  * User Identity/Role (e.g., "Student", "Developer")
  * Domain Preferences (e.g., "Prefers Python over Java", "Interested in AI ethics")
  * Fixed Facts (e.g., "Lives in Taipei", "Has a cat")
  * Stated Long-term Goals (e.g., "Wants to build a local agent")

- **CRITICAL: Distinguish between "Task Instructions" and "Persistent Traits":**
  * Do NOT record formatting instructions or tone requests for the CURRENT task (e.g., "be brief", "include links", "use a professional tone") as persistent memories.
  * ONLY record a communication preference if the user explicitly states it as a permanent rule (e.g., "I ALWAYS prefer brief answers" or "NEVER include links in your responses").
  * If the user says "For this question, be professional", do NOT save it.

- **Strict Exclusions (Do NOT save these as memories):**
  * Requests for specific output formats (Markdown tables, lists, code blocks).
  * Requests for links, citations, or references for a specific query.
  * Transient emotional states or one-off situational context.
  * Meta-talk about the current chat session.
  
Rules for UPDATING existing memories:
- Update a memory if the conversation reveals it is now outdated or inaccurate (e.g., user changed jobs, updated a preference, corrected a fact)
- Merge and store new information to one point based on topic so that it's easier to catch up instead of having a long list of fragmented entries.
- Consolidate multiple related memories into one updated entry when appropriate (update one, delete the others)
- Reference memories by their ID number

Rules for DELETING existing memories:
- Remove memories that are explicitly contradicted by the conversation
- Remove memories that are redundant (covered by another memory or a new/updated one)
- Remove memories that are no longer relevant or useful
- Reference memories by their ID number

Existing memories (with IDs for reference):
{existing_memories}

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{
    "new_memories": [
        {"section": "User Profile", "content": "fact about the user"}
    ],
    "updated_memories": [
        {"id": 0, "content": "corrected or refreshed fact"}
    ],
    "deleted_memory_ids": [3, 5]
}

Valid sections: "User Profile", "Preferences", "Key Facts", "Project Context"
If no changes needed, return: {"new_memories": [], "updated_memories": [], "deleted_memory_ids": []}"""

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


def _parse_pdf_reader_brief_payload(raw_text, filename):
    default_summary = f"Brief unavailable for {filename}."
    payload = {
        'summary': default_summary,
        'topics': [],
        'confidence': 0.45,
    }

    for candidate in reversed(_extract_json_objects(raw_text)):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue

        summary = str(
            parsed.get('summary')
            or parsed.get('brief')
            or parsed.get('result')
            or ''
        ).strip()
        if not summary:
            continue

        topics = parsed.get('key_topics')
        if not isinstance(topics, list):
            topics = parsed.get('topics') if isinstance(parsed.get('topics'), list) else []
        clean_topics = []
        for topic in topics[:8]:
            text = str(topic or '').strip()
            if text:
                clean_topics.append(text[:120])

        payload['summary'] = summary[:800]
        payload['topics'] = clean_topics
        payload['confidence'] = _clamp_confidence(parsed.get('confidence'), default=0.55)
        return payload

    raw_fallback = str(raw_text or '').strip()
    if raw_fallback:
        payload['summary'] = raw_fallback[:800]
    return payload


def _generate_pdf_upload_brief(filepath, filename):
    config = load_config()
    leader_model_id = str(config.get('Leader') or '').strip()
    pdf_reader_model_id = _resolve_pdf_reader_model_id(config, fallback_model_id=leader_model_id)

    system_prompt = (
        "You are PDF reader. Read the uploaded PDF and return only one JSON object with keys "
        "summary, key_topics, and confidence. "
        "summary must be 1-2 concise sentences. key_topics must be a short list of concrete topics."
    )
    user_prompt = json.dumps(
        {
            'task': 'Provide an upload-time brief so the Leader can decide whether to call PDF reader again later.',
            'filename': filename,
        },
        ensure_ascii=False,
    )

    raw = completion_response(
        model=pdf_reader_model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        chat_history=None,
        temperature=0.1,
        image_urls=None,
        pdf_inputs=[{'filename': filename, 'filepath': filepath}],
    )
    brief = _parse_pdf_reader_brief_payload(raw, filename)
    brief['model'] = pdf_reader_model_id
    brief['generated_at'] = datetime.now().isoformat()
    return brief


def _extract_document_payload(filepath, filename, pages_hint=None, pdf_brief_hint=None):
    """Extract normalized metadata/content for a stored document file."""
    file_size = os.path.getsize(filepath)
    file_ext = filename.lower().split('.')[-1]

    if file_ext == 'pdf':
        brief = pdf_brief_hint if isinstance(pdf_brief_hint, dict) else None
        if not brief or not str(brief.get('summary') or '').strip():
            try:
                brief = _generate_pdf_upload_brief(filepath, filename)
            except Exception as exc:
                raise ValueError(f'Failed to generate PDF brief for {filename}') from exc

        payload = {
            'filename': filename,
            'content': '[PDF uploaded]',
            'filepath': filepath,
            'pages': pages_hint,
            'size': file_size,
            'type': 'pdf',
            'pdf_brief_summary': str(brief.get('summary') or '').strip(),
            'pdf_brief_topics': list(brief.get('topics') or []),
            'pdf_brief_confidence': _clamp_confidence(brief.get('confidence'), default=0.45),
            'pdf_brief_model': str(brief.get('model') or ''),
            'pdf_brief_generated_at': str(brief.get('generated_at') or ''),
        }
        return payload

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
        'pdf_brief_summary': doc_payload.get('pdf_brief_summary'),
        'pdf_brief_topics': doc_payload.get('pdf_brief_topics'),
        'pdf_brief_confidence': doc_payload.get('pdf_brief_confidence'),
        'pdf_brief_model': doc_payload.get('pdf_brief_model'),
        'pdf_brief_generated_at': doc_payload.get('pdf_brief_generated_at')
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
            'pdf_brief_summary': item.get('pdf_brief_summary'),
            'pdf_brief_topics': item.get('pdf_brief_topics'),
            'pdf_brief_confidence': item.get('pdf_brief_confidence'),
            'pdf_brief_model': item.get('pdf_brief_model'),
            'pdf_brief_generated_at': item.get('pdf_brief_generated_at')
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
            pdf_brief_hint = None
            if str(doc_meta.get('type') or '').strip().lower() == 'pdf':
                pdf_brief_hint = {
                    'summary': doc_meta.get('pdf_brief_summary'),
                    'topics': doc_meta.get('pdf_brief_topics'),
                    'confidence': doc_meta.get('pdf_brief_confidence'),
                    'model': doc_meta.get('pdf_brief_model'),
                    'generated_at': doc_meta.get('pdf_brief_generated_at')
                }

            doc_payload = _extract_document_payload(
                filepath,
                filename,
                pages_hint=doc_meta.get('pages'),
                pdf_brief_hint=pdf_brief_hint
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
    """Get council role configuration with model details"""
    config = load_config()
    roles = {}
    for role_name in COUNCIL_ROLES:
        model_id = config.get(role_name, '')
        model_info = get_model_info(model_id)
        roles[role_name] = {
            'model_id': model_id,
            'model_name': model_info.get('name', model_id),
            'support_images': model_info.get('support_images', False),
            'support_pdf_input': model_info.get('support_pdf_input', False)
        }
    return jsonify(roles)


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
        for role_name in COUNCIL_ROLES:
            model_id = str(config.get(role_name, '') or '').strip()
            model_info = get_model_info(model_id) if model_id else {}
            role_rows.append({
                'role': role_name,
                'role_label': role_name.replace('_', ' '),
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
            pdf_brief_hint = None
            if str(doc_meta.get('type') or '').strip().lower() == 'pdf':
                pdf_brief_hint = {
                    'summary': doc_meta.get('pdf_brief_summary'),
                    'topics': doc_meta.get('pdf_brief_topics'),
                    'confidence': doc_meta.get('pdf_brief_confidence'),
                    'model': doc_meta.get('pdf_brief_model'),
                    'generated_at': doc_meta.get('pdf_brief_generated_at')
                }

            doc_payload = _extract_document_payload(
                filepath,
                filename,
                pages_hint=doc_meta.get('pages'),
                pdf_brief_hint=pdf_brief_hint
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

def extract_json_from_text(text):
    """Extract JSON object from text that might contain markdown code blocks or extra text"""
    if not isinstance(text, str):
        return ''

    stripped = text.strip()
    if not stripped:
        return ''

    # Try to find JSON in fenced code blocks first.
    codeblock_match = re.search(r'```(?:json)?\s*\n?(\{[\s\S]*?\})\s*```', stripped, re.IGNORECASE)
    if codeblock_match:
        return codeblock_match.group(1).strip()

    # Extract first balanced {...} object while respecting quoted strings.
    start = stripped.find('{')
    if start == -1:
        return ''  # No opening brace found - return empty instead of raw text

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return stripped[start:idx + 1]

    return ''  # Unbalanced braces - return empty


def _clamp_confidence(value, default=0.5):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _extract_json_objects(text):
    if not isinstance(text, str):
        return []

    stripped = text.strip()
    if not stripped:
        return []

    candidates = []
    in_string = False
    escape = False
    depth = 0
    start_idx = None

    for idx, ch in enumerate(stripped):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == '{':
            if depth == 0:
                start_idx = idx
            depth += 1
            continue

        if ch == '}':
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidates.append(stripped[start_idx:idx + 1])
                start_idx = None

    return candidates


def _normalize_script_name(value):
    raw = str(value or '').strip().replace('\\', '/')
    if not raw:
        return ''

    if '/' in raw:
        return ''

    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.py', raw):
        return ''

    return raw


def _normalize_local_tool_action(value):
    if not isinstance(value, dict):
        return None

    script_name = _normalize_script_name(value.get('script'))
    if not script_name:
        return None

    args = []
    raw_args = value.get('args', [])
    if isinstance(raw_args, list):
        for item in raw_args[:20]:
            if item is None:
                continue
            arg = str(item).strip()
            if arg:
                args.append(arg[:400])
    elif isinstance(raw_args, str):
        arg = raw_args.strip()
        if arg:
            args.append(arg[:400])

    return {
        'script': script_name,
        'args': args
    }


def _parse_agent_action_payload(raw_text):
    # Strip JSON blocks from raw text for fallback display
    cleaned_text = str(raw_text or '').strip()
    for json_candidate in _extract_json_objects(raw_text):
        cleaned_text = cleaned_text.replace(json_candidate, '').strip()
    
    fallback = {
        'thought': 'Direct final response (non-JSON).',
        'requires_verifier': False,
        'confidence': 0.35,
        'action': {
            'type': 'final_response',
            'text': cleaned_text or 'I could not generate a final answer this time.'
        }
    }

    for candidate in reversed(_extract_json_objects(raw_text)):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        action = payload.get('action')
        if not isinstance(action, dict):
            continue

        action_type = str(action.get('type') or '').strip().lower()
        if action_type not in {'skill_call', 'final_response'}:
            continue

        normalized = {
            'thought': str(payload.get('thought') or '').strip(),
            'requires_verifier': bool(payload.get('requires_verifier', False)),
            'confidence': _clamp_confidence(payload.get('confidence'), default=0.5),
            'action': {
                'type': action_type
            }
        }

        if action_type == 'skill_call':
            skill_id = str(action.get('skill') or '').strip()
            args = action.get('args') if isinstance(action.get('args'), dict) else {}
            if not skill_id:
                continue
            normalized['action']['skill'] = skill_id
            normalized['action']['args'] = args
            normalized['action']['reason'] = str(action.get('reason') or '').strip()
            local_tool = _normalize_local_tool_action(action.get('local_tool'))
            if local_tool:
                normalized['action']['local_tool'] = local_tool
        else:
            text = str(action.get('text') or '').strip()
            if not text:
                continue
            normalized['action']['text'] = text

        return normalized

    return fallback


def _build_skill_inventory_text(skills):
    lines = []
    for skill in skills:
        lines.append(
            f"- {skill.skill_id}: {skill.description}"
        )
    return '\n'.join(lines)


def _redact_sensitive_payload(value):
    sensitive_keys = {
        'api_key', 'apikey', 'authorization', 'token', 'access_token',
        'password', 'secret', 'file_data', 'bearer'
    }

    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            normalized_key = str(key or '').strip().lower()
            if normalized_key in sensitive_keys:
                redacted[key] = '[REDACTED]'
            else:
                redacted[key] = _redact_sensitive_payload(item)
        return redacted

    if isinstance(value, list):
        return [_redact_sensitive_payload(item) for item in value]

    if isinstance(value, str):
        text = value.strip()
        if re.search(r'(sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*)', text):
            return '[REDACTED]'
        if text.startswith('data:') and len(text) > 128:
            return '[REDACTED_BLOB]'
        return value

    return value


def _normalize_history_context_mode(value):
    normalized = str(value or '').strip().lower()
    if normalized in {'all', 'final_only'}:
        return normalized
    return 'final_only'


def _stringify_prompt_content(content):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    parts.append(str(item.get('text') or ''))
                elif item.get('type') == 'image_url':
                    parts.append('[image]')
                elif item.get('type') == 'file':
                    file_meta = item.get('file') if isinstance(item.get('file'), dict) else {}
                    parts.append(str(file_meta.get('filename') or '[file]'))
                else:
                    try:
                        parts.append(json.dumps(item, ensure_ascii=False))
                    except (TypeError, ValueError):
                        parts.append(str(item))
            else:
                parts.append(str(item))
        return '\n'.join(parts)
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def _estimate_prompt_tokens(system_prompt, user_prompt, chat_history):
    total_chars = len(_stringify_prompt_content(system_prompt))
    total_chars += len(_stringify_prompt_content(user_prompt))

    for message in (chat_history or []):
        if isinstance(message, dict):
            total_chars += len(str(message.get('role') or ''))
            total_chars += len(_stringify_prompt_content(message.get('content')))
        else:
            total_chars += len(_stringify_prompt_content(message))

    estimated = int(total_chars / APPROX_CHARS_PER_TOKEN) if total_chars > 0 else 0
    return max(0, estimated)


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

        if history_context_mode == 'final_only' and not _is_final_response_history_message(msg):
            continue

        raw = msg.get('raw_markdown')
        text = raw if isinstance(raw, str) and raw else msg.get('content', '')
        history.append({'role': 'assistant', 'content': text})

    return history


def _coerce_int(value, default_value, minimum=1):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default_value
    return max(minimum, parsed)


def _coerce_bool(value, default_value=True):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'1', 'true', 'yes', 'on'}:
            return True
        if lowered in {'0', 'false', 'no', 'off'}:
            return False
    return default_value


def _normalized_md_reader_config(config):
    md_config = config.get('md_reader', {}) if isinstance(config, dict) else {}
    if not isinstance(md_config, dict):
        md_config = {}

    merged = dict(DEFAULT_MD_READER_CONFIG)
    merged.update(md_config)
    merged['enabled'] = _coerce_bool(merged.get('enabled'), True)
    merged['max_inventory_files'] = _coerce_int(
        merged.get('max_inventory_files'),
        DEFAULT_MD_READER_CONFIG['max_inventory_files']
    )
    merged['preview_lines_per_file'] = _coerce_int(
        merged.get('preview_lines_per_file'),
        DEFAULT_MD_READER_CONFIG['preview_lines_per_file']
    )
    merged['preview_chars_per_file'] = _coerce_int(
        merged.get('preview_chars_per_file'),
        DEFAULT_MD_READER_CONFIG['preview_chars_per_file']
    )
    return merged


def _normalized_skills_config(config):
    skills_config = config.get('skills', {}) if isinstance(config, dict) else {}
    if not isinstance(skills_config, dict):
        skills_config = {}

    merged = dict(DEFAULT_SKILLS_CONFIG)
    merged.update(skills_config)
    merged['enabled'] = _coerce_bool(merged.get('enabled'), True)
    merged['folder'] = str(merged.get('folder') or 'skills').strip() or 'skills'
    merged['max_files'] = _coerce_int(merged.get('max_files'), DEFAULT_SKILLS_CONFIG['max_files'])
    merged['max_chars_per_file'] = _coerce_int(
        merged.get('max_chars_per_file'),
        DEFAULT_SKILLS_CONFIG['max_chars_per_file']
    )
    merged['max_total_chars'] = _coerce_int(
        merged.get('max_total_chars'),
        DEFAULT_SKILLS_CONFIG['max_total_chars']
    )
    return merged


def _resolve_skills_dir(config):
    skills_config = _normalized_skills_config(config)
    if not skills_config['enabled']:
        return '', '', {'status': 'disabled'}

    workspace_root = os.path.abspath('.')
    skills_folder = skills_config['folder']
    skills_dir = skills_folder if os.path.isabs(skills_folder) else os.path.join(workspace_root, skills_folder)
    skills_dir = os.path.abspath(skills_dir)

    if not skills_dir.startswith(workspace_root):
        return workspace_root, skills_dir, {'status': 'out_of_workspace'}

    if not os.path.isdir(skills_dir):
        return workspace_root, skills_dir, {'status': 'missing_folder'}

    return workspace_root, skills_dir, {'status': 'ok'}


def _collect_skills_markdown_files(config):
    workspace_root, skills_dir, dir_meta = _resolve_skills_dir(config)
    if dir_meta.get('status') != 'ok':
        return [], workspace_root, skills_dir, dir_meta

    markdown_paths = []
    for root, _, files in os.walk(skills_dir):
        for file_name in files:
            if file_name.lower().endswith('.md'):
                markdown_paths.append(os.path.join(root, file_name))
    markdown_paths.sort()

    if not markdown_paths:
        return [], workspace_root, skills_dir, {'status': 'empty_folder'}

    return markdown_paths, workspace_root, skills_dir, {'status': 'ok'}


def _extract_primary_h1_section(markdown_text):
    """Return content from the first H1 heading until the next H1, if present."""
    if not markdown_text:
        return ''

    lines = markdown_text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith('# '):
            start_idx = idx
            break

    if start_idx is None:
        return markdown_text

    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].lstrip().startswith('# '):
            end_idx = idx
            break

    return '\n'.join(lines[start_idx:end_idx]).strip()


def build_md_reader_inventory(config):
    """Prepare a bounded markdown file inventory for MD Reader selection."""
    markdown_paths, workspace_root, _, scan_meta = _collect_skills_markdown_files(config)
    if scan_meta.get('status') != 'ok':
        return '', {'status': scan_meta.get('status'), 'available_files': []}

    md_reader_config = _normalized_md_reader_config(config)
    limited_paths = markdown_paths[:md_reader_config['max_inventory_files']]

    lines = ["===== AVAILABLE MARKDOWN SKILLS INVENTORY ====="]
    available_files = []
    for path in limited_paths:
        relative_path = os.path.relpath(path, workspace_root).replace('\\', '/')
        available_files.append(relative_path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except OSError:
            continue

        primary_section = _extract_primary_h1_section(content)
        base_text = primary_section or content
        preview_lines = '\n'.join(base_text.splitlines()[:md_reader_config['preview_lines_per_file']])
        preview = preview_lines[:md_reader_config['preview_chars_per_file']]
        if len(preview_lines) > md_reader_config['preview_chars_per_file']:
            preview += "\n[... preview truncated ...]"

        lines.append(f"\n--- File: {relative_path} ---\n{preview}\n")

    return '\n'.join(lines), {
        'status': 'ok',
        'available_files': available_files
    }


def _parse_md_reader_payload(raw_text):
    fallback = {
        'selected_files': [],
        'reason': 'MD Reader output could not be parsed.'
    }
    if not raw_text:
        return fallback

    try:
        payload = json.loads(extract_json_from_text(raw_text))
    except Exception:
        return fallback

    selected = payload.get('selected_files', [])
    if not isinstance(selected, list):
        selected = []

    normalized_files = []
    for item in selected:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().replace('\\', '/')
        if cleaned and cleaned not in normalized_files:
            normalized_files.append(cleaned)

    reason = str(payload.get('reason') or '').strip() or fallback['reason']
    return {
        'selected_files': normalized_files,
        'reason': reason
    }


def _validate_selected_skill_files(config, selected_files):
    skills_config = _normalized_skills_config(config)
    markdown_paths, workspace_root, skills_dir, scan_meta = _collect_skills_markdown_files(config)
    if scan_meta.get('status') != 'ok':
        return [], {'status': scan_meta.get('status'), 'rejected_files': selected_files}

    available = {
        os.path.relpath(path, workspace_root).replace('\\', '/'): path
        for path in markdown_paths
    }

    validated_abs = []
    rejected = []
    for item in selected_files:
        absolute = available.get(item)
        if not absolute:
            rejected.append(item)
            continue
        if not os.path.abspath(absolute).startswith(os.path.abspath(skills_dir)):
            rejected.append(item)
            continue
        validated_abs.append(absolute)

    validated_abs = validated_abs[:skills_config['max_files']]
    validated_rel = [
        os.path.relpath(path, workspace_root).replace('\\', '/')
        for path in validated_abs
    ]
    return validated_rel, {
        'status': 'ok',
        'rejected_files': rejected
    }


def build_leader_skills_context_from_selected(config, selected_files):
    """Load selected markdown skill files for Leader prompts only."""
    skills_config = _normalized_skills_config(config)
    if not skills_config['enabled']:
        return '', {'status': 'disabled', 'loaded_files': []}

    markdown_paths, workspace_root, _, scan_meta = _collect_skills_markdown_files(config)
    if scan_meta.get('status') != 'ok':
        return '', {'status': scan_meta.get('status'), 'loaded_files': []}

    if not selected_files:
        return '', {'status': 'none_selected', 'loaded_files': []}

    path_map = {
        os.path.relpath(path, workspace_root).replace('\\', '/'): path
        for path in markdown_paths
    }
    selected_abs = [path_map[path] for path in selected_files if path in path_map]
    if not selected_abs:
        return '', {'status': 'none_loaded', 'loaded_files': []}

    sections = ["\n\n===== SELECTED SKILLS (FROM MARKREADER) ====="]
    total_chars = 0
    loaded_files = []
    for path in selected_abs:
        remaining = skills_config['max_total_chars'] - total_chars
        if remaining <= 0:
            break
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except OSError:
            continue
        if not content:
            continue

        snippet_limit = min(skills_config['max_chars_per_file'], remaining)
        snippet = content[:snippet_limit]
        if len(content) > snippet_limit:
            snippet += "\n[... skill content truncated ...]"
        relative_path = os.path.relpath(path, workspace_root).replace('\\', '/')
        sections.append(f"\n--- Skill File: {relative_path} ---\n{snippet}\n")
        loaded_files.append(relative_path)
        total_chars += len(snippet)

    if not loaded_files:
        return '', {'status': 'none_loaded', 'loaded_files': []}

    return ''.join(sections), {
        'status': 'loaded',
        'loaded_files': loaded_files
    }

def _normalize_requested_pdf_filenames(selected_pdf_filenames):
    normalized = []
    if isinstance(selected_pdf_filenames, str):
        selected_pdf_filenames = [selected_pdf_filenames]
    if not isinstance(selected_pdf_filenames, list):
        return normalized

    for item in selected_pdf_filenames[:20]:
        name = str(item or '').strip()
        if not name:
            continue
        if name not in normalized:
            normalized.append(name)
    return normalized


def build_document_context(
    conversation_id,
    system_prompt,
    support_images,
    support_pdf_input=False,
    user_query='',
    selected_pdf_filenames=None,
    attach_pdf_binary=False,
):
    """Build document context, image URLs, and optional scoped PDF binary inputs."""
    conv = get_conversation(conversation_id)
    documents = conv.get('uploaded_documents', {})
    image_urls = []
    pdf_inputs = []

    requested_names = _normalize_requested_pdf_filenames(selected_pdf_filenames)
    requested_set = set(requested_names)

    if not documents:
        return system_prompt, image_urls, pdf_inputs

    context_text = "\n\n===== AVAILABLE DOCUMENTS =====\n"
    for filename, doc_info in documents.items():
        doc_type = str(doc_info.get('type') or '').strip().lower()

        if doc_type == 'image':
            if support_images:
                img_base64 = encode_image_to_base64(doc_info.get('filepath'))
                if img_base64:
                    image_urls.append(img_base64)
                continue

            context_text += f"\n--- Image: {filename} ---\n"
            context_text += str(doc_info.get('content') or '[Image uploaded]')[:5000]
            if len(str(doc_info.get('content') or '')) > 5000:
                context_text += "\n[... content truncated ...]"
            context_text += "\n"
            continue

        if doc_type == 'pdf':
            context_text += f"\n--- PDF: {filename} ---\n"
            summary = str(doc_info.get('pdf_brief_summary') or '').strip()
            topics = doc_info.get('pdf_brief_topics') if isinstance(doc_info.get('pdf_brief_topics'), list) else []
            confidence = _clamp_confidence(doc_info.get('pdf_brief_confidence'), default=0.45)

            if summary:
                context_text += f"Brief: {summary}\n"
            else:
                context_text += "Brief: [No brief available yet.]\n"

            if topics:
                topic_text = ', '.join(str(topic).strip() for topic in topics[:8] if str(topic).strip())
                if topic_text:
                    context_text += f"Topics: {topic_text}\n"
            context_text += f"Brief confidence: {confidence:.2f}\n"

            should_attach = bool(attach_pdf_binary and support_pdf_input and doc_info.get('filepath'))
            if should_attach:
                if requested_set and filename not in requested_set:
                    continue
                pdf_inputs.append({'filename': filename, 'filepath': doc_info.get('filepath')})
            continue

        if doc_type == 'word':
            context_text += f"\n--- Word Document: {filename} ---\n"
            word_text = str(doc_info.get('text') or '').strip()
            if word_text:
                context_text += word_text[:12000]
                if len(word_text) > 12000:
                    context_text += "\n[... content truncated ...]"
            else:
                context_text += "[Word document uploaded, but no extractable text was found.]"
            context_text += "\n"
            continue

        context_text += f"\n--- Document: {filename} ---\n"
        context_text += str(doc_info.get('content') or '[Document uploaded]')[:5000]
        if len(str(doc_info.get('content') or '')) > 5000:
            context_text += "\n[... content truncated ...]"
        context_text += "\n"

    if context_text.strip() != '===== AVAILABLE DOCUMENTS =====':
        system_prompt += context_text

    return system_prompt, image_urls, pdf_inputs


def _extract_skill_pdf_filenames(skill_args):
    if not isinstance(skill_args, dict):
        return []
    raw = skill_args.get('filenames')
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    filenames = []
    for item in raw[:20]:
        name = str(item or '').strip()
        if not name:
            continue
        if name not in filenames:
            filenames.append(name)
    return filenames


def _resolve_selected_uploaded_pdf_filenames(conv, requested_filenames, default_to_all=False):
    available = _list_uploaded_pdf_filenames(conv)
    if not available:
        return []

    normalized_requested = _normalize_requested_pdf_filenames(requested_filenames)
    if normalized_requested:
        lookup = {name.lower(): name for name in available}
        selected = []
        for requested in normalized_requested:
            canonical = lookup.get(str(requested).lower())
            if canonical and canonical not in selected:
                selected.append(canonical)
        return selected

    if default_to_all:
        return list(available)

    return []

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
                    'pdf_brief_summary': doc.get('pdf_brief_summary'),
                    'pdf_brief_topics': doc.get('pdf_brief_topics'),
                    'pdf_brief_confidence': doc.get('pdf_brief_confidence'),
                    'pdf_brief_model': doc.get('pdf_brief_model'),
                    'pdf_brief_generated_at': doc.get('pdf_brief_generated_at')
                }
                for doc in conv.get('uploaded_documents', {}).values()
            ]
            if not is_saved:
                chat_data['isTemporary'] = True

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Snapshot save error for {session_id}: {e}")


def run_council_role(role_name, role_label, model_id, system_prompt, user_prompt, chat_history, conversation_id, run_group_id, support_images=False, support_pdf_input=False, on_stream_progress=None, _retry=False, _timeout_fallback_used=False, _fallback_used=False, internal_orchestration=False, stream_context=None):
    safe_stream_context = {
        key: value
        for key, value in dict(stream_context or {}).items()
        if value is not None
    }

    def emit_chat(event, payload=None):
        enriched = dict(payload or {})
        for key, value in safe_stream_context.items():
            enriched.setdefault(key, value)
        enriched['chat_id'] = conversation_id
        socketio.emit(event, enriched, to=conversation_id)

    """Run a single council role and stream its response"""
    try:
        conv = get_conversation(conversation_id)

        # Build document context
        system_prompt, image_urls, pdf_inputs = build_document_context(
            conversation_id,
            system_prompt,
            support_images,
            support_pdf_input=support_pdf_input,
            user_query=user_prompt
        )

        if internal_orchestration:
            timestamp = datetime.now().strftime("%H:%M:%S")
            emit_chat('council_status', {
                'role': role_name,
                'status': 'running',
                'run_group_id': run_group_id
            })

            raw_response = completion_response(
                model=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                chat_history=chat_history,
                temperature=0.7,
                image_urls=image_urls if support_images else None,
                pdf_inputs=pdf_inputs if support_pdf_input else None
            )
            converted_response = convert_to_traditional_chinese((raw_response or '').strip())
            if not converted_response:
                raise RuntimeError(f"{role_label} returned empty internal response")

            emit_chat('console_log', {'message': f"[{timestamp}] {role_label} completed (internal)"})
            emit_chat('council_status', {
                'role': role_name,
                'status': 'done',
                'run_group_id': run_group_id
            })

            return converted_response

        # Use a unique bot_id for the streaming UI
        bot_id = f"council-{role_name.lower()}-{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().strftime("%H:%M:%S")
        is_final_response = ('final response' in role_label.lower())

        pending_message_id = uuid.uuid4().hex
        append_conversation_message(
            conversation_id,
            role='assistant',
            content=f"[{role_label}] ",
            raw_markdown='',
            id=pending_message_id,
            bot_name=role_label,
            bot_id=bot_id,
            run_group_id=run_group_id,
            run_id=pending_message_id,
            role_name=role_name,
            model_id=model_id,
            thinking='',
            stream_status='running',
            is_final_response=is_final_response,
            **safe_stream_context
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
        timeout_before_first_chunk = False
        first_chunk_received = False
        last_snapshot_ts = 0.0
        estimated_input_tokens = _estimate_prompt_tokens(system_prompt, user_prompt, chat_history)
        first_chunk_timeout_seconds = DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS
        if estimated_input_tokens > LONG_INPUT_TOKEN_THRESHOLD:
            first_chunk_timeout_seconds = LONG_INPUT_FIRST_CHUNK_TIMEOUT_SECONDS
            emit_chat('console_log', {
                'message': (
                    f"[{timestamp}] {role_label} large prompt detected (~{estimated_input_tokens} tokens); "
                    f"first-chunk timeout extended to {int(first_chunk_timeout_seconds)}s"
                )
            })

        stream_queue = queue.Queue()
        stream_stop_event = threading.Event()
        stream_start_ts = time.time()

        def stream_worker():
            try:
                for chunk_type, chunk in completion_response_stream(
                    model=model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    chat_history=chat_history,
                    temperature=0.7,
                    image_urls=image_urls if support_images else None,
                    pdf_inputs=pdf_inputs if support_pdf_input else None
                ):
                    if stream_stop_event.is_set():
                        break
                    stream_queue.put(('chunk', chunk_type, chunk))
                stream_queue.put(('done', None, None))
            except Exception as exc:
                stream_queue.put(('error', exc, None))

        threading.Thread(target=stream_worker, daemon=True).start()

        while True:
            if not should_continue_streaming(conversation_id, pending_message_id):
                stopped_early = True
                stream_stop_event.set()
                break

            try:
                stream_event, chunk_type, chunk = stream_queue.get(timeout=0.25)
            except queue.Empty:
                if (
                    not first_chunk_received
                    and not _timeout_fallback_used
                    and (time.time() - stream_start_ts) >= first_chunk_timeout_seconds
                ):
                    timeout_before_first_chunk = True
                    stopped_early = True
                    stream_stop_event.set()
                    break
                continue

            if stream_event == 'done':
                break
            if stream_event == 'error':
                raise chunk_type

            first_chunk_received = True

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
                    f"[{role_label}] {full_response}"
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

        if timeout_before_first_chunk:
            timeout_ts = datetime.now().strftime("%H:%M:%S")
            timeout_notice = "[warning] Timed out before first stream chunk."
            emit_chat('console_log', {
                'message': (
                    f"[{timeout_ts}] {role_label} exceeded {int(first_chunk_timeout_seconds)}s "
                    "before first stream chunk"
                )
            })

            update_message_content(
                conversation_id,
                pending_message_id,
                f"[{role_label}] {timeout_notice}"
            )
            update_message_fields(
                conversation_id,
                pending_message_id,
                raw_markdown=timeout_notice,
                stream_status='timed_out',
                **safe_stream_context
            )
            emit_chat('ai_response_end', {
                'bot_name': role_label,
                'bot_id': bot_id,
                'message': timeout_notice,
                'thinking': '',
                'timestamp': timeout_ts,
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

            if conv.get('pending_message_id') == pending_message_id:
                conv['pending_message_id'] = None
            if on_stream_progress:
                on_stream_progress()

            fallback_model_id = str(load_config().get('FallBacker', '') or '').strip()
            can_fallback = (
                fallback_model_id
                and fallback_model_id != model_id
                and not _timeout_fallback_used
                and not _fallback_used
            )
            if can_fallback:
                fallback_label = _build_fallback_role_label(role_label, fallback_model_id)
                emit_chat('console_log', {
                    'message': f"[{timeout_ts}] Retrying {role_name.replace('_', ' ')} with FallBacker ({fallback_model_id})"
                })
                return run_council_role(
                    role_name=role_name,
                    role_label=fallback_label,
                    model_id=fallback_model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    chat_history=chat_history,
                    conversation_id=conversation_id,
                    run_group_id=run_group_id,
                    support_images=support_images,
                    support_pdf_input=support_pdf_input,
                    on_stream_progress=on_stream_progress,
                    _retry=False,
                    _timeout_fallback_used=True,
                    _fallback_used=True,
                    internal_orchestration=internal_orchestration,
                    stream_context=safe_stream_context
                )
            return ''

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
            update_message_content(conversation_id, pending_message_id, f"[{role_label}] {partial_converted}")
            update_message_fields(
                conversation_id,
                pending_message_id,
                thinking=thinking_converted,
                raw_markdown=partial_converted,
                stream_status='stopped',
                **safe_stream_context
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
        update_message_content(conversation_id, pending_message_id, f"[{role_label}] {converted_response}")

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
            raw_markdown=converted_response,
            stream_status='done',
            **safe_stream_context
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
                chat_history, conversation_id, run_group_id,
                support_images=support_images,
                support_pdf_input=support_pdf_input,
                on_stream_progress=on_stream_progress,
                _retry=True,
                _timeout_fallback_used=_timeout_fallback_used,
                _fallback_used=_fallback_used,
                internal_orchestration=internal_orchestration,
                stream_context=safe_stream_context
            )

        fallback_model_id = str(load_config().get('FallBacker', '') or '').strip()
        can_fallback = (
            fallback_model_id
            and fallback_model_id != model_id
            and not _fallback_used
        )
        if can_fallback:
            fallback_label = _build_fallback_role_label(role_label, fallback_model_id)
            emit_chat('console_log', {
                'message': f"[{timestamp}] {role_name.replace('_', ' ')} switching to FallBacker ({fallback_model_id}) after failed response"
            })
            return run_council_role(
                role_name=role_name,
                role_label=fallback_label,
                model_id=fallback_model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                chat_history=chat_history,
                conversation_id=conversation_id,
                run_group_id=run_group_id,
                support_images=support_images,
                support_pdf_input=support_pdf_input,
                on_stream_progress=on_stream_progress,
                _retry=False,
                _timeout_fallback_used=_timeout_fallback_used,
                _fallback_used=True,
                internal_orchestration=internal_orchestration,
                stream_context=safe_stream_context
            )

        # Second failure: skip this role
        notice = f"{role_label} has encountered an error and has to go without it."
        emit_chat('council_status', {'role': role_name, 'status': 'error'})
        bot_id = f"council-{role_name.lower()}"
        emit_chat('ai_response', {
            'bot_name': role_label,
            'bot_id': bot_id,
            'message': f"[warning] {notice}",
            'timestamp': timestamp
        })
        return None



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
    attachments = _coerce_message_attachments(data.get('attachments'))
    restored_docs = _rehydrate_uploaded_documents_from_attachments(conv, attachments)
    if restored_docs > 0:
        emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Restored {restored_docs} uploaded source(s) for this run",
            'chat_id': conversation_id
        }, to=conversation_id)

    if conv.get('is_generating'):
        redirect_message = str(data.get('message') or '').strip()
        if redirect_message:
            conv['agent_redirect_message'] = redirect_message
            emit('console_log', {
                'message': f"[{datetime.now().strftime('%H:%M:%S')}] Redirect instruction queued for active run",
                'chat_id': conversation_id
            }, to=conversation_id)
            socketio.emit('agent_step', {
                'chat_id': conversation_id,
                'run_group_id': conv.get('current_run_group_id'),
                'step_type': 'redirect_input',
                'status': 'queued',
                'summary': redirect_message
            }, to=conversation_id)
            return
        emit('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Generation already running for this chat",
            'chat_id': conversation_id
        }, to=conversation_id)
        return

    conv['abort_event'].clear()
    conv['is_generating'] = True
    socketio.start_background_task(handle_message_task, data, conversation_id)


def _emit_agent_step(emit_chat, run_group_id, step_type, status, summary, payload=None, iteration=None):
    event_payload = {
        'run_group_id': run_group_id,
        'step_type': step_type,
        'status': status,
        'summary': str(summary or '')
    }
    if iteration is not None:
        event_payload['iteration'] = int(iteration)
    if payload is not None:
        event_payload['payload'] = _redact_sensitive_payload(payload)
    emit_chat('agent_step', event_payload)


def _parse_skill_result_payload(raw_text):
    fallback = {
        'result': str(raw_text or '').strip(),
        'confidence': 0.4,
        'notes': []
    }

    for candidate in reversed(_extract_json_objects(raw_text)):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        result = str(payload.get('result') or '').strip()
        if not result:
            continue

        notes = payload.get('notes')
        if not isinstance(notes, list):
            notes = []

        return {
            'result': result,
            'confidence': _clamp_confidence(payload.get('confidence'), default=0.5),
            'notes': [str(item) for item in notes]
        }

    return fallback


def _run_memory_management_agent(config, model_id, user_message, final_response):
    result = {
        'added': 0,
        'updated': 0,
        'deleted': 0,
        'payload': {}
    }

    if not (memory_manager.is_enabled(config) and memory_manager.auto_extract_enabled(config)):
        return result

    existing_flat = memory_manager.read_flat(config)
    existing_lines = []
    for entry in existing_flat:
        existing_lines.append(f"[ID {entry['id']}] ({entry['section']}) {entry['content']}")
    existing_text = '\n'.join(existing_lines) if existing_lines else '(none yet)'

    extract_sys = MEMORY_EXTRACT_PROMPT.replace('{existing_memories}', existing_text)
    extract_user = f"User: {user_message}\nLeader: {str(final_response or '')}"

    extract_raw = completion_response(
        model=model_id,
        system_prompt=extract_sys,
        user_prompt=extract_user,
        temperature=0.2
    )
    extracted_json = extract_json_from_text(extract_raw)
    payload = json.loads(extracted_json) if extracted_json else {}
    result['payload'] = payload

    updated_items = payload.get('updated_memories', [])
    deleted_ids = payload.get('deleted_memory_ids', [])
    new_memories = payload.get('new_memories', [])

    if isinstance(updated_items, list):
        for item in updated_items:
            if not isinstance(item, dict):
                continue
            mem_id = item.get('id')
            content = str(item.get('content') or '').strip()
            if mem_id is None or not content:
                continue
            try:
                memory_manager.update_memory(int(mem_id), content, config)
                result['updated'] += 1
            except Exception:
                continue

    if isinstance(deleted_ids, list):
        for mem_id in sorted(deleted_ids, reverse=True):
            try:
                memory_manager.delete_memory(int(mem_id), config)
                result['deleted'] += 1
            except Exception:
                continue

    if isinstance(new_memories, list) and new_memories:
        before = len(memory_manager.read_flat(config))
        memory_manager.add_memories_bulk(new_memories, config)
        after = len(memory_manager.read_flat(config))
        result['added'] += max(0, after - before)

    return result


def _build_model_document_inputs(
    conversation_id,
    model_id,
    system_prompt,
    user_query='',
    selected_pdf_filenames=None,
    attach_pdf_binary=False,
):
    """Attach uploaded document context and optional scoped PDF binary inputs."""
    model_info = get_model_info(model_id)
    support_images = bool(model_info.get('support_images', False))
    support_pdf_input = bool(model_info.get('support_pdf_input', False))
    return build_document_context(
        conversation_id,
        system_prompt,
        support_images=support_images,
        support_pdf_input=support_pdf_input,
        user_query=user_query,
        selected_pdf_filenames=selected_pdf_filenames,
        attach_pdf_binary=bool(attach_pdf_binary and support_pdf_input)
    )


def _completion_response_with_doc_fallback(
    *,
    model,
    system_prompt,
    user_prompt,
    chat_history=None,
    temperature=0.2,
    image_urls=None,
    pdf_inputs=None
):
    """Run model completion with graceful retries when multimodal payloads are rejected."""
    try:
        return completion_response(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            chat_history=chat_history,
            temperature=temperature,
            image_urls=image_urls,
            pdf_inputs=pdf_inputs
        )
    except Exception as first_exc:
        logger.warning(
            "Agent completion failed with document inputs (model=%s, image_inputs=%s, pdf_inputs=%s): %s",
            model,
            bool(image_urls),
            bool(pdf_inputs),
            str(first_exc)
        )
        if pdf_inputs:
            try:
                logger.warning("Retrying agent completion without PDF binary inputs (model=%s)", model)
                return completion_response(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    chat_history=chat_history,
                    temperature=temperature,
                    image_urls=image_urls,
                    pdf_inputs=None
                )
            except Exception:
                pass

        if image_urls:
            try:
                logger.warning("Retrying agent completion without image/PDF binary inputs (model=%s)", model)
                return completion_response(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    chat_history=chat_history,
                    temperature=temperature,
                    image_urls=None,
                    pdf_inputs=None
                )
            except Exception:
                pass

        raise first_exc


def _run_agent_single_leader_workflow(
    *,
    conv,
    config,
    user_message,
    user_system_prompt,
    chat_history,
    conversation_id,
    run_group_id,
    emit_chat,
    auto_save_chat,
    stop_if_aborted
):
    leader_model_id = str(config.get('Leader') or '').strip()
    skills_cfg = config.get('skills', {}) if isinstance(config.get('skills', {}), dict) else {}
    skills_root = str(skills_cfg.get('folder') or 'skills').strip() or 'skills'
    allow_legacy_flat = bool(skills_cfg.get('allow_legacy_flat', True))
    discovered_skills = skill_registry.discover_skills(skills_root, allow_legacy_flat=allow_legacy_flat)

    skill_catalog = skill_registry.build_skill_catalog(discovered_skills)

    _emit_agent_step(
        emit_chat,
        run_group_id,
        'skills_catalog',
        'ready',
        f'Discovered {len(discovered_skills)} skills',
        payload={'skills': skill_catalog}
    )

    if not discovered_skills:
        return "No skills were discovered. Please add skills/<name>/SKILL.md files."

    memory_context = memory_manager.build_memory_context(config)
    _emit_agent_step(
        emit_chat,
        run_group_id,
        'memory_read',
        'done',
        'Loaded persistent memory context',
        payload={'memory_context': memory_context}
    )

    model_map = skills_cfg.get('model_map', {}) if isinstance(skills_cfg.get('model_map', {}), dict) else {}
    warning_interval = int((config.get('agent_loop', {}) or {}).get('warning_interval', 10) or 10)
    warning_interval = max(1, warning_interval)

    loop_observations = []
    iteration = 0

    while True:
        if stop_if_aborted():
            return ''

        redirect_message = str(conv.get('agent_redirect_message') or '').strip()
        if redirect_message:
            conv['agent_redirect_message'] = ''
            loop_observations.append(f"[redirect] {redirect_message}")
            _emit_agent_step(
                emit_chat,
                run_group_id,
                'redirect_input',
                'applied',
                redirect_message,
                iteration=iteration
            )

        if iteration > 0 and (iteration % warning_interval) == 0:
            _emit_agent_step(
                emit_chat,
                run_group_id,
                'iteration_warning',
                'notice',
                f'Iteration {iteration} reached configured warning interval ({warning_interval})',
                iteration=iteration
            )

        skill_inventory_text = _build_skill_inventory_text(discovered_skills)
        context_lines = [
            'User request:',
            user_message,
            '',
            'Available skills:',
            skill_inventory_text,
            '',
            'Observed results so far:'
        ]
        context_lines.extend(loop_observations[-12:] if loop_observations else ['(none yet)'])
        context_text = '\n'.join(context_lines)

        planner_system = LEADER_AGENT_ACTION_PROMPT
        if memory_context:
            planner_system += f"\n\n{memory_context}"
        if user_system_prompt:
            planner_system = f"{user_system_prompt}\n\n{planner_system}"

        planner_system, planner_image_urls, planner_pdf_inputs = _build_model_document_inputs(
            conversation_id,
            leader_model_id,
            planner_system,
            user_query=user_message
        )

        planner_raw = _completion_response_with_doc_fallback(
            model=leader_model_id,
            system_prompt=planner_system,
            user_prompt=context_text,
            chat_history=chat_history,
            temperature=0.2,
            image_urls=planner_image_urls or None,
            pdf_inputs=planner_pdf_inputs or None
        )
        action_payload = _parse_agent_action_payload(planner_raw)

        _emit_agent_step(
            emit_chat,
            run_group_id,
            'leader_plan',
            'done',
            action_payload.get('thought') or 'Plan generated',
            payload=action_payload,
            iteration=iteration
        )

        action = action_payload.get('action', {})
        action_type = str(action.get('type') or '').strip().lower()

        if action_type == 'final_response':
            final_text = str(action.get('text') or '').strip()
            if not final_text:
                final_text = 'I could not generate a complete answer yet.'

            try:
                memory_model_id = _resolve_lite_model_id(config, leader_model_id)
                memory_result = _run_memory_management_agent(
                    config,
                    memory_model_id,
                    user_message,
                    final_text
                )
                _emit_agent_step(
                    emit_chat,
                    run_group_id,
                    'memory_write',
                    'done',
                    f"Memory updated (+{memory_result['added']} ~{memory_result['updated']} -{memory_result['deleted']})",
                    payload=memory_result
                )
            except Exception as exc:
                _emit_agent_step(
                    emit_chat,
                    run_group_id,
                    'memory_write',
                    'error',
                    f'Memory update failed: {str(exc)}',
                    payload={'error': str(exc)}
                )

            return final_text

        if action_type != 'skill_call':
            loop_observations.append('[planner] Invalid action type returned; retrying.')
            iteration += 1
            continue

        skill_id = str(action.get('skill') or '').strip()
        skill_args = action.get('args') if isinstance(action.get('args'), dict) else {}
        reason = str(action.get('reason') or '').strip()
        local_tool = action.get('local_tool') if isinstance(action.get('local_tool'), dict) else None

        selected_pdf_filenames = []
        if str(skill_id).strip().lower() == PDF_READER_SKILL_ID:
            requested_pdf_filenames = _extract_skill_pdf_filenames(skill_args)
            selected_pdf_filenames = _resolve_selected_uploaded_pdf_filenames(
                conv,
                requested_filenames=requested_pdf_filenames,
                default_to_all=False
            )
            if not selected_pdf_filenames:
                loop_observations.append(
                    '[skill-error] pdfer-skill requires valid args.filenames matching uploaded PDFs.'
                )
                _emit_agent_step(
                    emit_chat,
                    run_group_id,
                    'skill_call',
                    'error',
                    'pdfer-skill requires valid filenames from uploaded PDFs',
                    payload={'skill': skill_id, 'args': skill_args},
                    iteration=iteration
                )
                iteration += 1
                continue

            skill_args = dict(skill_args)
            skill_args['filenames'] = selected_pdf_filenames

        skill_def = skill_registry.get_skill_by_id(discovered_skills, skill_id)
        if not skill_def:
            loop_observations.append(f"[skill-error] Skill not found: {skill_id}")
            _emit_agent_step(
                emit_chat,
                run_group_id,
                'skill_call',
                'error',
                f'Skill not found: {skill_id}',
                payload={'skill': skill_id, 'args': skill_args},
                iteration=iteration
            )
            iteration += 1
            continue

        skill_model_id = str(model_map.get(skill_def.skill_id) or leader_model_id).strip() or leader_model_id
        tool_result = None

        _emit_agent_step(
            emit_chat,
            run_group_id,
            'skill_call',
            'running',
            f"Calling {skill_def.skill_id}",
            payload={
                'skill': skill_def.skill_id,
                'reason': reason,
                'args': skill_args,
                'model': skill_model_id,
                'local_tool': local_tool
            },
            iteration=iteration
        )

        if local_tool:
            _emit_agent_step(
                emit_chat,
                run_group_id,
                'skill_tool',
                'running',
                f"Running {skill_def.skill_id}/{local_tool.get('script')}",
                payload={'skill': skill_def.skill_id, 'local_tool': local_tool},
                iteration=iteration
            )
            tool_result = skill_tool_runner.run_skill_script(
                skill_file_path=skill_def.path,
                script_name=local_tool.get('script'),
                args=local_tool.get('args', [])
            )
            tool_status = 'done' if tool_result.get('ok') else 'error'
            _emit_agent_step(
                emit_chat,
                run_group_id,
                'skill_tool',
                tool_status,
                f"{skill_def.skill_id} tool {tool_status}",
                payload=tool_result,
                iteration=iteration
            )
            if tool_result.get('ok'):
                loop_observations.append(
                    f"[tool:{skill_def.skill_id}] {local_tool.get('script')} succeeded"
                )
            else:
                loop_observations.append(
                    f"[tool-error:{skill_def.skill_id}] {tool_result.get('error', 'tool failed')}"
                )

        skill_system_prompt = SKILL_EXECUTION_PROMPT_TEMPLATE.format(
            skill_id=skill_def.skill_id,
            skill_description=skill_def.description,
            skill_content=skill_def.content
        )
        attach_pdf_binary = bool(skill_def.skill_id == PDF_READER_SKILL_ID)
        skill_system_prompt, skill_image_urls, skill_pdf_inputs = _build_model_document_inputs(
            conversation_id,
            skill_model_id,
            skill_system_prompt,
            user_query=str(skill_args.get('task') or user_message),
            selected_pdf_filenames=selected_pdf_filenames,
            attach_pdf_binary=attach_pdf_binary
        )
        skill_user_prompt = json.dumps({
            'task': skill_args.get('task') or user_message,
            'args': skill_args,
            'reason': reason,
            'local_tool_result': tool_result
        }, ensure_ascii=False, indent=2)

        skill_result_raw = _completion_response_with_doc_fallback(
            model=skill_model_id,
            system_prompt=skill_system_prompt,
            user_prompt=skill_user_prompt,
            chat_history=None,
            temperature=0.2,
            image_urls=skill_image_urls or None,
            pdf_inputs=skill_pdf_inputs or None
        )
        parsed_skill_result = _parse_skill_result_payload(skill_result_raw)

        _emit_agent_step(
            emit_chat,
            run_group_id,
            'skill_call',
            'done',
            f"{skill_def.skill_id} completed",
            payload={
                'skill': skill_def.skill_id,
                'args': skill_args,
                'result': parsed_skill_result,
                'model': skill_model_id
            },
            iteration=iteration
        )

        loop_observations.append(
            f"[skill:{skill_def.skill_id}] {parsed_skill_result.get('result', '')[:800]}"
        )

        iteration += 1


def handle_message_task(data, conversation_id):
    """Handle user message and run the council workflow"""
    user_message = data.get('message', '')
    user_system_prompt = data.get('system_prompt', '')
    existing_user_message_id = str(data.get('existing_user_message_id') or '').strip()

    def emit_chat(event, payload=None):
        enriched = dict(payload or {})
        enriched['chat_id'] = conversation_id
        socketio.emit(event, enriched, to=conversation_id)
            
    def auto_save_chat():
        persist_chat_snapshot(conversation_id, user_system_prompt)

    def finalize_generation():
        conv['is_generating'] = False
        conv['pending_message_id'] = None

    def complete_workflow(include_run_group_end=True):
        emit_chat('all_done')
        if include_run_group_end:
            emit_chat('run_group_end', {
                'run_group_id': run_group_id,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        conv['current_run_group_id'] = None
        finalize_generation()

    def stop_if_aborted():
        if conv['abort_event'].is_set():
            complete_workflow(include_run_group_end=False)
            return True
        return False

    conv = get_conversation(conversation_id)
    run_group_id = _build_run_group_id(conversation_id)
    conv['current_run_group_id'] = run_group_id
    current_user_index = None

    if existing_user_message_id:
        for idx, msg in enumerate(conv.get('messages', [])):
            if msg.get('id') == existing_user_message_id and msg.get('role') == 'user':
                current_user_index = idx
                existing_raw = msg.get('raw_markdown')
                user_message = existing_raw if isinstance(existing_raw, str) and existing_raw else msg.get('content', '')
                msg['run_group_id'] = run_group_id
                break

    if current_user_index is None:
        appended_user = append_conversation_message(
            conversation_id,
            "user",
            user_message,
            raw_markdown=user_message,
            run_group_id=run_group_id
        )
        current_user_index = len(conv.get('messages', [])) - 1
        existing_user_message_id = appended_user.get('id', '')

    auto_save_chat()

    emit_chat('run_group_start', {
        'run_group_id': run_group_id,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })

    # Load council config
    config = load_config()
    history_context_mode = _normalize_history_context_mode(config.get('history_context_mode', 'final_only'))

    # Build chat history (exclude current user message) using configured prompt policy.
    chat_history = _build_prompt_chat_history(
        conv.get('messages', []),
        current_user_index,
        history_context_mode
    )

    timestamp = datetime.now().strftime("%H:%M:%S")
    emit_chat('console_log', {
        'message': (
            f"[{timestamp}] Leader-agent workflow started "
            f"(history: {history_context_mode})"
        )
    })

    try:
        final_response = _run_agent_single_leader_workflow(
            conv=conv,
            config=config,
            user_message=user_message,
            user_system_prompt=user_system_prompt,
            chat_history=chat_history,
            conversation_id=conversation_id,
            run_group_id=run_group_id,
            emit_chat=emit_chat,
            auto_save_chat=auto_save_chat,
            stop_if_aborted=stop_if_aborted
        )
    except Exception as exc:
        _log_internal_error('leader agent workflow failed', exc)
        final_response = 'I encountered an internal error while running the agent workflow.'

    if stop_if_aborted():
        return

    final_response = str(final_response or '').strip()
    if not final_response:
        final_response = 'I could not generate a final answer this time.'

    final_message_id = uuid.uuid4().hex
    final_bot_id = f"agent-leader-final-{uuid.uuid4().hex[:8]}"
    final_model_id = str(config.get('Leader') or '')
    final_label = f"Leader - Final Response ({final_model_id.strip()})"
    final_ts = datetime.now().strftime("%H:%M:%S")

    try:
        append_conversation_message(
            conversation_id,
            role='assistant',
            content=f"[{final_label}] {final_response}",
            raw_markdown=final_response,
            id=final_message_id,
            bot_name=final_label,
            bot_id=final_bot_id,
            run_group_id=run_group_id,
            run_id=final_message_id,
            role_name='Leader',
            model_id=final_model_id,
            thinking='',
            stream_status='done',
            is_final_response=True
        )

        emit_chat('ai_response_start', {
            'bot_name': final_label,
            'bot_id': final_bot_id,
            'timestamp': final_ts,
            'message_id': final_message_id,
            'run_id': final_message_id,
            'run_group_id': run_group_id,
            'role_name': 'Leader',
            'model_id': final_model_id,
            'is_final_response': True
        })
        emit_chat('ai_response_end', {
            'bot_name': final_label,
            'bot_id': final_bot_id,
            'message': final_response,
            'thinking': '',
            'timestamp': final_ts,
            'message_id': final_message_id,
            'run_id': final_message_id,
            'run_group_id': run_group_id,
            'role_name': 'Leader',
            'model_id': final_model_id,
            'is_final_response': True
        })
    except Exception as exc:
        _log_internal_error('leader final response emit failed', exc)
        emit_chat('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Final response emit fallback engaged"
        })
        emit_chat('ai_response_end', {
            'bot_name': final_label,
            'bot_id': final_bot_id,
            'message': final_response,
            'thinking': '',
            'timestamp': final_ts,
            'message_id': final_message_id,
            'run_id': final_message_id,
            'run_group_id': run_group_id,
            'role_name': 'Leader',
            'model_id': final_model_id,
            'is_final_response': True
        })

    emit_chat('council_status', {
        'role': 'Leader',
        'status': 'done',
        'run_group_id': run_group_id
    })

    auto_save_chat()
    complete_workflow(include_run_group_end=True)
    return

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
