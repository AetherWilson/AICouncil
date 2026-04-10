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
from services import memory_manager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHAT_HISTORY_FOLDER'] = 'chat_history'
app.config['TEMP_CHAT_HISTORY_FOLDER'] = 'temp_chat_history'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")
logger = logging.getLogger(__name__)

COUNCIL_ROLES = ['MarkReader', 'Leader', 'Researcher', 'Creator', 'Analyzer', 'Verifier']
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
#     active_stream_task: handle,
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
        'active_stream_task': None,
        'pending_message_id': None,
        'abort_event': threading.Event(),
        'uploaded_documents': {},
        'run_group_counter': 0,
        'current_run_group_id': None
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

# Council role system prompts
LEADER_DISTRIBUTE_PROMPT = """You are the Leader of an AI council. Your only job is task routing.

Role boundary policy (HARD CONSTRAINTS):
1) Researcher
- Owns: web/info gathering, source lookup, evidence collection, factual freshness checks.
- Must be assigned for informational requests that depend on facts, real-world knowledge, explanations, comparisons, recommendations, summaries, or any knowledge the Leader may know only partially.
- If an answer could be incomplete, stale, memory-based, or only partially known, assign Researcher even when the Leader feels highly confident.
- Leader confidence is NEVER a valid reason to skip Researcher on knowledge-dependent requests.
- Forbidden: mathematical derivation, UX design planning, creative writing, auditing teammates.

2) Creator
- Owns: creative ideation, UX/content planning, writing high-quality user-facing text.
- Forbidden: fact-checking, source validation, logical auditing, questioning teammate correctness.

3) Analyzer
- Owns: formal logic, mathematical derivation, quantitative reasoning.
- Forbidden: web research, source verification, teammate auditing.

4) Verifier
- Owns: auditing other roles' outputs for factual errors, logical flaws, unsupported claims, and step validity.
- Forbidden: producing primary creative/research/math deliverables except minimal corrections.

Critical routing rules:
- ONLY Verifier may evaluate, doubt, or critique other roles.
- Never assign cross-role audit tasks to Researcher, Creator, or Analyzer.
- If the user request mixes domains, split into parallel role-specific tasks.
- If the user is asking for information, facts, explanation, comparison, recommendation, summary, or any real-world/domain knowledge, assign Researcher unless the task is purely about transforming user-provided content.
- When in doubt about whether the Leader's own knowledge is complete enough, assign Researcher.
- Skip Researcher ONLY for clearly non-factual tasks such as rewriting, formatting, translating, style transformation, or purely creative work that does not depend on external facts.
- Tasks must be concrete, output-oriented, and minimal (no vague wording).
- If a role is unnecessary, set that role's task to "SKIP".
- If any non-Verifier role is assigned, Verifier should usually be assigned to check that output.
- Use "direct_response" only when the request is simple AND non-knowledge-dependent; if the answer depends on facts the Leader may know only partially, do not use direct_response and assign Researcher instead.
- Never provide user-facing advice in "analysis" or any *_task field.
- "analysis" must describe routing intent only (short, internal).
- User-facing answer text is allowed only in "direct_response", and only when all role tasks are "SKIP".

Return ONLY a valid JSON object in this exact format (no markdown, no extra text):
{
  "analysis": "Brief routing analysis",
  "researcher_task": "Concrete task for Researcher, or SKIP",
    "creator_task": "Concrete task for Creator, or SKIP",
  "analyzer_task": "Concrete task for Analyzer, or SKIP",
    "verifier_task": "Concrete verification task for Verifier, or SKIP",
    "direct_response": "Final answer to user when request is simple; otherwise empty string"
}

Hard constraints reminder:
- Output must be a single JSON object only.
- Do not include markdown code fences.
- Do not place any final answer text in "analysis".
"""

RESEARCHER_PROMPT = """You are the Researcher in an AI council. Provide the most detailed, useful research possible for the assigned task by default.

- Be thorough and comprehensive unless the Leader's assigned task explicitly asks for brevity, a short summary, or a narrower scope
- Include all directly relevant facts, context, caveats, definitions, comparisons, and supporting details that help produce a stronger final answer
- Prefer depth over brevity when there is any doubt
- Cite URLs where available
- Organize clearly using markdown
- Do not add filler, but do not omit helpful detail merely to stay short

Do NOT prefix your response with your role name."""

CREATOR_PROMPT = """You are the Creator in an AI council. Produce concise, high-quality creative content for the task.

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
- Preserve the information needed to justify the conclusion, not just the conclusion itself
- Keep key evidence when relevant: source links, factual support, assumptions, caveats, and the calculation steps needed to understand or verify the result
- Prefer a strong structure: direct answer first, then supporting detail underneath when needed
- Do not over-compress Researcher or Analyzer output if that would remove essential support for the answer
- Use proper markdown formatting
- Never use raw HTML tags (for example: <p>, <ul>, <li>, <div>); use markdown syntax only

If the task depends on evidence or reasoning, the final answer should normally include short supporting sections such as:
- "Why / Basis"
- "Calculation" or "Reasoning"
- "Sources"
- "Caveats"

Respond directly — do NOT mention the council, team members, or that you are combining results. Do NOT prefix with your role name."""

LEADER_FAST_PROMPT = """You are the Leader of an AI council operating in FAST mode.

Return a direct answer to the user in a single pass.

Rules:
- Do not delegate to other roles
- Be concise but complete
- Use markdown formatting only (no raw HTML tags)
- If uncertainty is material, state assumptions clearly

Respond directly. Do NOT prefix with your role name."""

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

VERIFIER_DEBATE_STEP_A_PROMPT = """You are the Verifier in a post-round debate stage.

Your job in this step is to raise focused doubts about the first-round outputs.

Rules:
- Keep everything minimal and sufficient only
- Question only concrete, high-impact issues that could materially change the final answer
- Prefer fewer, stronger doubts over many small ones
- If there is no issue worth debating, return empty questioned_roles and empty doubt_points
- Use lowercase role names: researcher, creator, analyzer
- Rank doubts by impact: critical > important > minor
- For each doubt, specify: what is wrong, why it matters, and what evidence, source, or calculation is needed to resolve it
- Avoid vague language like "unclear" or "needs more detail"
- Only question if you can articulate the specific problem
- Focus especially on unsupported claims, missing source support, broken logic, hidden assumptions, or calculation steps that do not support the stated result

Return ONLY valid JSON in this exact shape:
{
    "questioned_roles": ["researcher", "analyzer"],
    "doubt_points": [
        {
            "point_id": "P1",
            "target_role": "researcher",
            "doubt": "Specific concise doubt in 1-2 sentences",
            "severity": "critical",
            "required_evidence": "cite source or provide calculation"
        }
    ],
    "confidence_in_doubt": 0.0
}

Do not include markdown or any extra text."""

CALLOUT_REBUTTAL_PROMPT = """You are in a targeted rebuttal step.

You were explicitly questioned by the Verifier.

Rules:
- Answer only the questioned points
- Keep the response as short as possible while fully resolving each doubt
- Prefer direct correction or direct evidence over long explanation
- If the Verifier requested a source, cite the source briefly
- If the Verifier questioned a calculation or assumption, show only the steps needed to validate or revise the claim
- Return ONLY valid JSON in this exact shape:
{
    "rebuttals": [
        {
            "point_id": "P1",
            "status": "refute",
            "evidence": "specific data/logic",
            "updated_claim": "revised statement if needed"
        }
    ]
}
- Include one rebuttal object for every questioned point_id
- You may admit, correct, refute, or provide brief supporting evidence
- Allowed status values: "acknowledge" | "refute" | "clarify"
- Keep evidence and updated_claim concise but specific
- Do not include markdown or any extra text
"""

VERIFIER_DEBATE_STEP_C_PROMPT = """You are the Verifier in a post-round debate stage.

Your job in this step is to judge the rebuttal for the questioned role.

Rules:
- Keep output minimal and sufficient only
- Judge based only on the provided doubts and rebuttal
- Use one final verdict for this role in this cycle
- Prefer ending the debate once the material issue is resolved
- Continue only if a specific unresolved problem remains that could still change the final answer

Return ONLY valid JSON in this exact shape:
{
    "verdict": "accept",
    "reason": "Short reason with only the decisive point",
    "next_action": "enough_for_this_role",
    "updated_confidence": 0.0
}

Allowed values:
- verdict: "accept" | "partially_accept" | "reject"
- next_action: "continue_question" | "move_to_other_point" | "enough_for_this_role"

Do not include markdown or any extra text."""

DEBATE_MAX_CYCLES = 3

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


def _extract_document_payload(filepath, filename, pages_hint=None):
    """Extract normalized metadata/content for a stored document file."""
    file_size = os.path.getsize(filepath)
    file_ext = filename.lower().split('.')[-1]

    if file_ext == 'pdf':
        return {
            'filename': filename,
            'content': '[PDF uploaded]',
            'filepath': filepath,
            'pages': pages_hint,
            'size': file_size,
            'type': 'pdf'
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

    live_conv = conversations.get(normalized_chat_id)
    if live_conv and (live_conv.get('messages') or live_conv.get('is_generating')):
        chat_data['messages'] = _history_to_ui_messages(normalized_chat_id)
        chat_data['uploadedDocuments'] = _conversation_documents_to_ui(normalized_chat_id)

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

    live_conv = conversations.get(normalized_chat_id)
    if live_conv and (live_conv.get('messages') or live_conv.get('is_generating')):
        chat_data['messages'] = _history_to_ui_messages(normalized_chat_id)
        chat_data['uploadedDocuments'] = _conversation_documents_to_ui(normalized_chat_id)

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


def _normalize_questioned_role(role_name):
    normalized = (role_name or '').strip().lower()
    mapping = {
        'researcher': 'Researcher',
        'creator': 'Creator',
        'analyzer': 'Analyzer'
    }
    return mapping.get(normalized)


def _is_skip_task(task_value):
    if not isinstance(task_value, str):
        return False
    return task_value.strip().upper() == 'SKIP'


def _normalize_workflow_mode(value):
    normalized = str(value or '').strip().lower()
    if normalized in {'auto', 'fast', 'heavy'}:
        return normalized
    return 'auto'


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


def _normalize_task_distribution(tasks):
    normalized = dict(tasks) if isinstance(tasks, dict) else {}

    # Ensure expected keys exist for downstream logic.
    for key in ('researcher_task', 'creator_task', 'analyzer_task', 'verifier_task'):
        if key not in normalized:
            normalized[key] = 'SKIP'

    direct_response = normalized.get('direct_response', '')
    if direct_response is None:
        direct_response = ''
    elif not isinstance(direct_response, str):
        direct_response = str(direct_response)
    normalized['direct_response'] = direct_response.strip()

    return normalized


def _build_safe_default_task_distribution():
    return _normalize_task_distribution({
        'analysis': 'Fallback route: leader task JSON unavailable.',
        'researcher_task': 'SKIP',
        'creator_task': 'Create the primary answer that directly solves the user request in concise markdown.',
        'analyzer_task': 'SKIP',
        'verifier_task': 'Audit the Creator output for factual/logical issues and report only concrete problems.',
        'direct_response': ''
    })


def _parse_task_distribution_payload(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError('Task distribution response is empty')

    json_text = extract_json_from_text(raw_text)
    if not isinstance(json_text, str) or not json_text.strip():
        raise ValueError('No JSON content found in task distribution response')

    trimmed = json_text.strip()
    if not trimmed.startswith('{'):
        raise ValueError('Task distribution response did not contain a JSON object')

    payload = json.loads(trimmed)
    if not isinstance(payload, dict):
        raise ValueError('Task distribution payload must be a JSON object')

    return _normalize_task_distribution(payload)


def _parse_step_a_payload(raw_text):
    fallback = {
        'questioned_roles': [],
        'doubt_points': [],
        'confidence_in_doubt': 0.5
    }
    if not raw_text:
        return fallback

    try:
        payload = json.loads(extract_json_from_text(raw_text))
    except Exception:
        return fallback

    questioned_roles = []
    for role in payload.get('questioned_roles', []):
        normalized = _normalize_questioned_role(role)
        if normalized and normalized not in questioned_roles:
            questioned_roles.append(normalized)

    doubt_points = []
    for item in payload.get('doubt_points', []):
        if not isinstance(item, dict):
            continue
        target_role = _normalize_questioned_role(item.get('target_role'))
        point_id = str(item.get('point_id') or '').strip()
        doubt = str(item.get('doubt') or '').strip()
        severity = str(item.get('severity') or '').strip().lower() or 'important'
        required_evidence = str(item.get('required_evidence') or '').strip()
        if not target_role or not doubt:
            continue
        if not point_id:
            point_id = f"P{len(doubt_points) + 1}"
        if severity not in {'critical', 'important', 'minor'}:
            severity = 'important'
        doubt_points.append({
            'point_id': point_id,
            'target_role': target_role,
            'doubt': doubt,
            'severity': severity,
            'required_evidence': required_evidence
        })
        if target_role not in questioned_roles:
            questioned_roles.append(target_role)

    return {
        'questioned_roles': questioned_roles,
        'doubt_points': doubt_points,
        'confidence_in_doubt': _clamp_confidence(payload.get('confidence_in_doubt'), default=0.5)
    }


def _parse_step_c_payload(raw_text):
    fallback = {
        'verdict': 'accept',
        'reason': 'Unable to parse verifier verdict cleanly.',
        'next_action': 'enough_for_this_role',
        'updated_confidence': 0.5
    }
    if not raw_text:
        return fallback

    try:
        payload = json.loads(extract_json_from_text(raw_text))
    except Exception:
        return fallback

    verdict = str(payload.get('verdict') or '').strip().lower()
    if verdict not in {'accept', 'partially_accept', 'reject'}:
        verdict = fallback['verdict']

    next_action = str(payload.get('next_action') or '').strip().lower()
    if next_action not in {'continue_question', 'move_to_other_point', 'enough_for_this_role'}:
        next_action = fallback['next_action']

    reason = str(payload.get('reason') or '').strip() or fallback['reason']

    return {
        'verdict': verdict,
        'reason': reason,
        'next_action': next_action,
        'updated_confidence': _clamp_confidence(payload.get('updated_confidence'), default=0.5)
    }


def _parse_step_a_payload_strict(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    json_text = extract_json_from_text(raw_text)
    if not json_text:
        return None
    try:
        payload = json.loads(json_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    required_keys = ['questioned_roles', 'doubt_points', 'confidence_in_doubt']
    if any(key not in payload for key in required_keys):
        return None
    return _parse_step_a_payload(json.dumps(payload, ensure_ascii=False))


def _parse_step_c_payload_strict(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    json_text = extract_json_from_text(raw_text)
    if not json_text:
        return None
    try:
        payload = json.loads(json_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    required_keys = ['verdict', 'reason', 'next_action', 'updated_confidence']
    if any(key not in payload for key in required_keys):
        return None
    return _parse_step_c_payload(json.dumps(payload, ensure_ascii=False))


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

def build_document_context(conversation_id, system_prompt, support_images, support_pdf_input=False):
    """Build document context, image URLs, and PDF inputs for a council role."""
    conv = get_conversation(conversation_id)
    documents = conv.get('uploaded_documents', {})
    image_urls = []
    pdf_inputs = []
    if documents:
        context_text = ""
        unsupported_pdf_filenames = []
        for filename, doc_info in documents.items():
            doc_type = doc_info.get('type')

            if doc_type == 'image':
                if support_images:
                    img_base64 = encode_image_to_base64(doc_info['filepath'])
                    if img_base64:
                        image_urls.append(img_base64)
                    continue

                if not context_text:
                    context_text = "\n\n===== AVAILABLE DOCUMENTS =====\n"
                context_text += f"\n--- Image: {filename} ---\n"
                context_text += doc_info.get('content', '[Image uploaded]')[:5000]
                if len(doc_info.get('content', '')) > 5000:
                    context_text += "\n[... content truncated ...]"
                context_text += "\n"
                continue

            if doc_type == 'pdf':
                if support_pdf_input and doc_info.get('filepath'):
                    pdf_inputs.append({
                        'filename': filename,
                        'filepath': doc_info['filepath']
                    })
                    continue

                unsupported_pdf_filenames.append(filename)
                continue

            if not context_text:
                context_text = "\n\n===== AVAILABLE DOCUMENTS =====\n"

            if doc_type == 'word':
                context_text += f"\n--- Word Document: {filename} ---\n"
                word_text = doc_info.get('text', '').strip()
                if word_text:
                    context_text += word_text[:12000]
                    if len(word_text) > 12000:
                        context_text += "\n[... content truncated ...]"
                else:
                    context_text += "[Word document uploaded, but no extractable text was found.]"
                context_text += "\n"
                continue

            context_text += f"\n--- Document: {filename} ---\n"
            context_text += doc_info.get('content', '[Document uploaded]')[:5000]
            if len(doc_info.get('content', '')) > 5000:
                context_text += "\n[... content truncated ...]"
            context_text += "\n"

        if context_text:
            system_prompt += context_text

        if unsupported_pdf_filenames:
            joined = ', '.join(unsupported_pdf_filenames)
            system_prompt += (
                "\n\n===== PDF ACCESS LIMITATION =====\n"
                "The user attached PDF files, but this model cannot read PDF inputs directly.\n"
                f"Attached PDF filenames: {joined}\n"
                "Do not claim to have read the PDF contents. Ask the user to switch to a PDF-capable model if needed.\n"
            )

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
            support_pdf_input=support_pdf_input
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
            timeout_notice = "⚠️ Timed out before first stream chunk."
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
    if _is_skip_task(role_task):
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
        support_pdf_input=role_info.get('support_pdf_input', False),
        on_stream_progress=auto_save_chat
    )
    if role_response is not None:
        council_results[role_name] = role_response
        auto_save_chat()
    return role_response


def _build_debate_round_snapshot(available_targets, role_latest_outputs):
    sections = []
    for role_name in (available_targets or []):
        sections.append(
            f"\n\n===== {role_name.replace('_', ' ')} Current Output =====\n"
            f"{role_latest_outputs.get(role_name, '')}"
        )
    return ''.join(sections)


def _build_debate_history_snapshot(debate_records, role_filter=None):
    relevant_records = [
        record for record in (debate_records or [])
        if not role_filter or record.get('role') == role_filter
    ]
    if not relevant_records:
        return ''

    history_snapshot = "\n\n===== Prior Debate Records ====="
    for record in relevant_records:
        history_snapshot += (
            f"\nCycle {record['cycle']} | {record['role'].replace('_', ' ')} | "
            f"verdict={record['verdict']} | next_action={record['next_action']}"
            f"\nReason: {record['reason']}"
        )
    return history_snapshot


def _build_step_c_verdict_context(cycle, role_name, role_points, rebuttal,
                                  role_latest_outputs, available_targets,
                                  debate_records, step_a_payload):
    round_snapshot = _build_debate_round_snapshot(available_targets, role_latest_outputs)
    role_history_snapshot = _build_debate_history_snapshot(debate_records, role_filter=role_name)
    global_history_snapshot = _build_debate_history_snapshot(debate_records)
    role_output = role_latest_outputs.get(role_name, '')
    role_history_count = len([
        record for record in (debate_records or [])
        if record.get('role') == role_name
    ])

    context_meta = {
        'has_role_output': bool(str(role_output).strip()),
        'role_history_count': role_history_count,
        'global_history_count': len(debate_records or []),
        'question_count': len(role_points or []),
        'confidence_in_doubt': _clamp_confidence(
            (step_a_payload or {}).get('confidence_in_doubt'),
            default=0.5
        )
    }

    context_block = (
        "\n\nDebate verdict context policy:"
        "\nAll required context for this verdict is provided below."
        "\nDo not ask for additional internal state, prior cycles, or system logs."
        f"\n\nCycle: {cycle}"
        f"\nRole under review: {role_name}"
        f"\nQuestion count for this role: {context_meta['question_count']}"
        f"\nVerifier confidence_in_doubt from Step A: {context_meta['confidence_in_doubt']}"
        f"\n\n{role_name} output under review:\n{role_output}"
        f"\n\nQuestioned points:\n{json.dumps(role_points, ensure_ascii=False, indent=2)}"
        f"\n\nRebuttal:\n{rebuttal}"
        + (f"\n\n{role_name} debate history:{role_history_snapshot}" if role_history_snapshot else '')
        + (f"\n\nGlobal debate history:{global_history_snapshot}" if global_history_snapshot else '')
        + (f"\n\nCurrent round outputs:{round_snapshot}" if round_snapshot else '')
    )
    return context_block, context_meta


def _run_post_round_debate_stage(user_message, user_system_prompt, tasks, config, council_results,
                                 chat_history, conversation_id, run_group_id, auto_save_chat,
                                 emit_chat, stop_if_aborted):
    """Run a verifier-led debate stage after the first round without changing first-round steps."""
    verifier_task = tasks.get('verifier_task', 'SKIP')
    if _is_skip_task(verifier_task):
        return ''

    debate_targets = ['Researcher', 'Creator', 'Analyzer']
    available_targets = [role for role in debate_targets if council_results.get(role)]
    if not available_targets:
        return ''

    verifier_model_id = config.get('Verifier', '')
    verifier_info = get_model_info(verifier_model_id)
    debate_records = []
    role_latest_outputs = {role: council_results.get(role, '') for role in debate_targets}

    for cycle in range(1, DEBATE_MAX_CYCLES + 1):
        if stop_if_aborted():
            return ''

        cycle_ts = datetime.now().strftime("%H:%M:%S")
        emit_chat('console_log', {
            'message': f"[{cycle_ts}] Debate cycle {cycle}/{DEBATE_MAX_CYCLES}: Verifier questioning..."
        })

        round_snapshot = _build_debate_round_snapshot(available_targets, role_latest_outputs)
        history_snapshot = _build_debate_history_snapshot(debate_records)

        step_a_system = (
            VERIFIER_DEBATE_STEP_A_PROMPT
            + f"\n\nOriginal user request: {user_message}"
            + f"\n\nDebate objective: {verifier_task}"
            + f"\n\nRound 0 outputs:{round_snapshot}"
            + history_snapshot
        )
        if user_system_prompt:
            step_a_system = user_system_prompt + "\n\n" + step_a_system

        step_a_raw = run_council_role(
            role_name='Verifier',
            role_label=f'Verifier - Debate Questioning C{cycle} ({verifier_model_id})',
            model_id=verifier_model_id,
            system_prompt=step_a_system,
            user_prompt='Return the questioning JSON for this debate cycle.',
            chat_history=chat_history,
            conversation_id=conversation_id,
            run_group_id=run_group_id,
            support_images=verifier_info.get('support_images', False),
            support_pdf_input=verifier_info.get('support_pdf_input', False),
            on_stream_progress=auto_save_chat,
            stream_context={
                'event_kind': 'verifier_questioning',
                'debate_cycle': cycle
            }
        )
        if stop_if_aborted():
            return ''

        step_a_payload = _parse_step_a_payload_strict(step_a_raw)
        if step_a_payload is None:
            emit_chat('console_log', {
                'message': (
                    f"[{datetime.now().strftime('%H:%M:%S')}] Debate cycle {cycle}: "
                    "Step-A JSON parse failed; retrying with strict schema instruction."
                )
            })
            strict_step_a_system = (
                step_a_system
                + "\n\nSTRICT RETRY INSTRUCTION: Return exactly one valid JSON object in the required schema."
                + " Do not include markdown, code fences, natural language explanation, or extra keys."
            )
            try:
                step_a_retry_raw = completion_response(
                    model=verifier_model_id,
                    system_prompt=strict_step_a_system,
                    user_prompt='Return ONLY the questioning JSON now.',
                    chat_history=chat_history,
                    temperature=0.2
                )
                step_a_payload = _parse_step_a_payload_strict(step_a_retry_raw)
            except Exception as exc:
                _log_internal_error('debate step-a strict retry failed', exc)
                step_a_payload = None

        if step_a_payload is None:
            emit_chat('console_log', {
                'message': (
                    f"[{datetime.now().strftime('%H:%M:%S')}] Debate cycle {cycle}: "
                    "Step-A parse failed after strict retry; debate closed safely to avoid false assumptions."
                )
            })
            break

        questioned_roles = [
            role for role in step_a_payload.get('questioned_roles', [])
            if role in available_targets
        ]
        doubt_points = [
            point for point in step_a_payload.get('doubt_points', [])
            if point.get('target_role') in questioned_roles
        ]

        if not questioned_roles or not doubt_points:
            emit_chat('console_log', {
                'message': f"[{datetime.now().strftime('%H:%M:%S')}] Debate cycle {cycle}: no further doubts, debate closed."
            })
            break

        high_priority_points = [
            point for point in doubt_points
            if point.get('severity') in {'critical', 'important'}
        ]
        if not high_priority_points and step_a_payload.get('confidence_in_doubt', 0.0) < 0.6:
            emit_chat('console_log', {
                'message': f"[{datetime.now().strftime('%H:%M:%S')}] Debate cycle {cycle}: only low-confidence minor doubts remained, debate closed."
            })
            break

        role_rebuttals = {}
        for role_name in questioned_roles:
            if stop_if_aborted():
                return ''

            role_points = [p for p in doubt_points if p.get('target_role') == role_name]
            if not role_points:
                continue

            model_id = config.get(role_name, '')
            role_info = get_model_info(model_id)
            points_json = json.dumps(role_points, ensure_ascii=False, indent=2)

            rebuttal_system = (
                CALLOUT_REBUTTAL_PROMPT
                + f"\n\nOriginal user request: {user_message}"
                + f"\n\nYour current output:\n{role_latest_outputs.get(role_name, '')}"
                + f"\n\nVerifier doubts (full detail):\n{points_json}"
            )
            if user_system_prompt:
                rebuttal_system = user_system_prompt + "\n\n" + rebuttal_system

            rebuttal_response = run_council_role(
                role_name=role_name,
                role_label=f"{role_name.replace('_', ' ')} - Debate Rebuttal C{cycle} ({model_id})",
                model_id=model_id,
                system_prompt=rebuttal_system,
                user_prompt='Respond to all doubts briefly and directly.',
                chat_history=chat_history,
                conversation_id=conversation_id,
                run_group_id=run_group_id,
                support_images=role_info.get('support_images', False),
                support_pdf_input=role_info.get('support_pdf_input', False),
                on_stream_progress=auto_save_chat,
                stream_context={
                    'event_kind': 'debate_rebuttal',
                    'target_role': role_name,
                    'debate_cycle': cycle
                }
            )
            if rebuttal_response is None:
                continue

            role_rebuttals[role_name] = rebuttal_response
            role_latest_outputs[role_name] = rebuttal_response
            auto_save_chat()

        if not role_rebuttals:
            emit_chat('console_log', {
                'message': f"[{datetime.now().strftime('%H:%M:%S')}] Debate cycle {cycle}: no rebuttal received, debate closed."
            })
            break

        should_continue = False
        for role_name, rebuttal in role_rebuttals.items():
            if stop_if_aborted():
                return ''

            role_points = [p for p in doubt_points if p.get('target_role') == role_name]
            step_c_context, step_c_meta = _build_step_c_verdict_context(
                cycle=cycle,
                role_name=role_name,
                role_points=role_points,
                rebuttal=rebuttal,
                role_latest_outputs=role_latest_outputs,
                available_targets=available_targets,
                debate_records=debate_records,
                step_a_payload=step_a_payload
            )
            emit_chat('console_log', {
                'message': (
                    f"[{datetime.now().strftime('%H:%M:%S')}] Debate cycle {cycle}: "
                    f"Verdict context for {role_name} "
                    f"(has_role_output={step_c_meta['has_role_output']}, "
                    f"role_history={step_c_meta['role_history_count']}, "
                    f"global_history={step_c_meta['global_history_count']}, "
                    f"question_count={step_c_meta['question_count']})."
                )
            })
            step_c_system = (
                VERIFIER_DEBATE_STEP_C_PROMPT
                + f"\n\nOriginal user request: {user_message}"
                + step_c_context
            )
            if user_system_prompt:
                step_c_system = user_system_prompt + "\n\n" + step_c_system

            step_c_raw = run_council_role(
                role_name='Verifier',
                role_label=f'Verifier - Debate Verdict C{cycle} {role_name.replace("_", " ")} ({verifier_model_id})',
                model_id=verifier_model_id,
                system_prompt=step_c_system,
                user_prompt='Return the verdict JSON for this role in this cycle.',
                chat_history=chat_history,
                conversation_id=conversation_id,
                run_group_id=run_group_id,
                support_images=verifier_info.get('support_images', False),
                support_pdf_input=verifier_info.get('support_pdf_input', False),
                on_stream_progress=auto_save_chat,
                stream_context={
                    'event_kind': 'verifier_verdict',
                    'target_role': role_name,
                    'debate_cycle': cycle
                }
            )
            verdict = _parse_step_c_payload_strict(step_c_raw)
            if verdict is None:
                emit_chat('console_log', {
                    'message': (
                        f"[{datetime.now().strftime('%H:%M:%S')}] Debate cycle {cycle}: "
                        f"Step-C JSON parse failed for {role_name}; retrying with strict schema instruction."
                    )
                })
                strict_step_c_system = (
                    step_c_system
                    + "\n\nSTRICT RETRY INSTRUCTION: Return exactly one valid JSON object in the required schema."
                    + " Do not include markdown, code fences, natural language explanation, or extra keys."
                )
                try:
                    step_c_retry_raw = completion_response(
                        model=verifier_model_id,
                        system_prompt=strict_step_c_system,
                        user_prompt='Return ONLY the verdict JSON now.',
                        chat_history=chat_history,
                        temperature=0.2
                    )
                    verdict = _parse_step_c_payload_strict(step_c_retry_raw)
                except Exception as exc:
                    _log_internal_error('debate step-c strict retry failed', exc)
                    verdict = None

            if verdict is None:
                verdict = {
                    'verdict': 'reject',
                    'reason': 'Verifier verdict JSON parse failed after strict retry; treated as unresolved for safety.',
                    'next_action': 'continue_question',
                    'updated_confidence': 0.0
                }

            verdict_record = {
                'cycle': cycle,
                'role': role_name,
                'doubt_points': role_points,
                'rebuttal': rebuttal,
                **verdict
            }
            debate_records.append(verdict_record)

            if verdict['next_action'] in {'continue_question', 'move_to_other_point'} and verdict['verdict'] != 'accept':
                should_continue = True

        if not should_continue:
            emit_chat('console_log', {
                'message': f"[{datetime.now().strftime('%H:%M:%S')}] Debate cycle {cycle}: resolved, debate closed."
            })
            break

    if not debate_records:
        return ''

    for role_name, rebuttal in role_latest_outputs.items():
        if rebuttal:
            council_results[role_name] = rebuttal

    summary_text = "\n\n===== Debate Stage Summary ====="
    for record in debate_records:
        summary_text += (
            f"\n\n[Cycle {record['cycle']}] {record['role'].replace('_', ' ')}"
            f"\nDoubts: {json.dumps(record['doubt_points'], ensure_ascii=False)}"
            f"\nRebuttal: {record['rebuttal']}"
            f"\nVerdict: {record['verdict']}"
            f"\nReason: {record['reason']}"
            f"\nNext action: {record['next_action']}"
            f"\nUpdated confidence: {record['updated_confidence']}"
        )

    council_results['Verifier_Debate'] = summary_text.strip()
    auto_save_chat()
    return summary_text

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
    existing_user_message_id = str(data.get('existing_user_message_id') or '').strip()
    workflow_mode = _normalize_workflow_mode(data.get('workflow_mode'))
    
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

    def run_memory_management(council_results_local=None, direct_response_text=''):
        # Expose memory extraction/editing as a visible thinking-process stream.
        try:
            if not (memory_manager.is_enabled(config) and memory_manager.auto_extract_enabled(config)):
                return

            mem_writer_model_id = config.get('MemWriter', '') or leader_model_id
            timestamp = datetime.now().strftime("%H:%M:%S")
            message_id = uuid.uuid4().hex
            bot_id = f"council-memwriter-{uuid.uuid4().hex[:8]}"
            bot_name = f"MemWriter - Memory Management ({mem_writer_model_id})"
            thinking_chunks = []

            def emit_memory_thinking(text):
                chunk = str(text or '')
                if not chunk:
                    return
                if not chunk.endswith('\n'):
                    chunk += '\n'
                thinking_chunks.append(chunk)
                emit_chat('ai_thinking_chunk', {
                    'bot_id': bot_id,
                    'chunk': chunk,
                    'message_id': message_id,
                    'run_id': message_id,
                    'run_group_id': run_group_id,
                    'role_name': 'MemWriter',
                    'model_id': mem_writer_model_id,
                    'is_final_response': False,
                    'event_kind': 'memory_management'
                })

            def finalize_memory_stream(output_text, status='done'):
                output = str(output_text or '').strip() or 'Memory management completed.'
                thinking_text = ''.join(thinking_chunks).strip()

                update_message_content(
                    conversation_id,
                    message_id,
                    f"[{bot_name}] {output}"
                )
                update_message_fields(
                    conversation_id,
                    message_id,
                    raw_markdown=output,
                    thinking=thinking_text,
                    stream_status=status,
                    role_name='MemWriter',
                    model_id=mem_writer_model_id,
                    event_kind='memory_management'
                )

                emit_chat('ai_response_end', {
                    'bot_name': bot_name,
                    'bot_id': bot_id,
                    'message': output,
                    'thinking': thinking_text,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'message_id': message_id,
                    'run_id': message_id,
                    'run_group_id': run_group_id,
                    'role_name': 'MemWriter',
                    'model_id': mem_writer_model_id,
                    'is_final_response': False,
                    'event_kind': 'memory_management'
                })

                if conv.get('pending_message_id') == message_id:
                    conv['pending_message_id'] = None

                auto_save_chat()

            append_conversation_message(
                conversation_id,
                role='assistant',
                content=f"[{bot_name}] ",
                raw_markdown='',
                id=message_id,
                bot_name=bot_name,
                bot_id=bot_id,
                run_group_id=run_group_id,
                run_id=message_id,
                role_name='MemWriter',
                model_id=mem_writer_model_id,
                thinking='',
                stream_status='running',
                is_final_response=False,
                event_kind='memory_management'
            )
            conv['pending_message_id'] = message_id

            emit_chat('ai_response_start', {
                'bot_name': bot_name,
                'bot_id': bot_id,
                'timestamp': timestamp,
                'message_id': message_id,
                'run_id': message_id,
                'run_group_id': run_group_id,
                'role_name': 'MemWriter',
                'model_id': mem_writer_model_id,
                'is_final_response': False,
                'event_kind': 'memory_management'
            })

            emit_memory_thinking('Preparing memory extraction context.')

            # Build existing memories with IDs so the model can reference them.
            existing_flat = memory_manager.read_flat(config)
            emit_memory_thinking(f"Loaded {len(existing_flat)} existing memories.")

            if existing_flat:
                existing_lines = []
                for entry in existing_flat:
                    existing_lines.append(f"  [ID {entry['id']}] ({entry['section']}) {entry['content']}")
                existing_text = '\n'.join(existing_lines)
            else:
                existing_text = '(none yet)'

            extract_sys = MEMORY_EXTRACT_PROMPT.replace('{existing_memories}', existing_text)
            convo_summary = f"User: {user_message}"

            if direct_response_text:
                convo_summary += f"\nLeader: {direct_response_text[:500]}"

            if isinstance(council_results_local, dict):
                for key in ['Researcher', 'Creator', 'Analyzer', 'Verifier']:
                    if key in council_results_local:
                        convo_summary += f"\n{key}: {council_results_local[key][:500]}"

            emit_memory_thinking('Requesting MemWriter memory-edit proposal.')

            try:
                extract_raw = completion_response(
                    model=mem_writer_model_id,
                    system_prompt=extract_sys,
                    user_prompt=convo_summary,
                    temperature=0.3
                )
                extracted_json_str = extract_json_from_text(extract_raw)
                if not extracted_json_str or not extracted_json_str.strip():
                    _log_internal_error('run_memory_management empty extractor response', 
                                      f"extract_raw: {repr(extract_raw[:200])}\nextracted_json_str: {repr(extracted_json_str[:200] if extracted_json_str else '')}")
                    emit_chat('console_log', {
                        'message': f"[{datetime.now().strftime('%H:%M:%S')}] Memory extraction skipped: empty extractor output"
                    })
                    emit_memory_thinking('Extractor returned empty payload; no edits applied.')
                    finalize_memory_stream('Memory extraction skipped: empty extractor output.', status='error')
                    return
                extract_payload = json.loads(extracted_json_str)
            except json.JSONDecodeError as exc:
                _log_internal_error('run_memory_management invalid extractor JSON', exc)
                emit_chat('console_log', {
                    'message': f"[{datetime.now().strftime('%H:%M:%S')}] Memory extraction skipped: invalid extractor output"
                })
                emit_memory_thinking('Extractor returned invalid JSON; no edits applied.')
                finalize_memory_stream('Memory extraction skipped: invalid extractor output.', status='error')
                return
            except Exception as exc:
                _log_internal_error('run_memory_management extractor request failed', exc)
                emit_chat('console_log', {
                    'message': f"[{datetime.now().strftime('%H:%M:%S')}] Memory extraction skipped: extractor unavailable"
                })
                emit_memory_thinking('Extractor request failed; no edits applied.')
                finalize_memory_stream('Memory extraction skipped: extractor unavailable.', status='error')
                return

            try:
                added = 0
                updated = 0
                deleted = 0
                applied_updates = []
                applied_deletes = []

                def _preview_text(value, limit=160):
                    cleaned = ' '.join(str(value or '').split())
                    if len(cleaned) <= limit:
                        return cleaned
                    return cleaned[:limit - 3] + '...'

                def _build_counts(flat_entries):
                    counts = {}
                    for entry in (flat_entries or []):
                        section = str(entry.get('section', 'Key Facts') or 'Key Facts').strip() or 'Key Facts'
                        content = str(entry.get('content', '') or '').strip()
                        if not content:
                            continue
                        key = (section, content)
                        counts[key] = counts.get(key, 0) + 1
                    return counts

                def _decrement_count(counts, key):
                    current = int(counts.get(key, 0) or 0)
                    if current <= 0:
                        return
                    if current == 1:
                        counts.pop(key, None)
                    else:
                        counts[key] = current - 1

                existing_by_id = {}
                for entry in existing_flat:
                    try:
                        existing_by_id[int(entry.get('id'))] = {
                            'section': str(entry.get('section', 'Key Facts') or 'Key Facts').strip() or 'Key Facts',
                            'content': str(entry.get('content', '') or '').strip()
                        }
                    except (TypeError, ValueError):
                        continue

                updated_memories = extract_payload.get('updated_memories', [])
                deleted_ids = extract_payload.get('deleted_memory_ids', [])
                new_memories = extract_payload.get('new_memories', [])

                emit_memory_thinking(
                    f"Extractor proposed changes: +{len(new_memories) if isinstance(new_memories, list) else 0} "
                    f"new, ~{len(updated_memories) if isinstance(updated_memories, list) else 0} "
                    f"updates, -{len(deleted_ids) if isinstance(deleted_ids, list) else 0} deletes."
                )

                # Process updates first (updates don't shift indices).
                if isinstance(updated_memories, list):
                    for item in updated_memories:
                        if not isinstance(item, dict):
                            continue
                        mid = item.get('id')
                        content = item.get('content', '').strip()
                        if mid is None or not content:
                            continue
                        try:
                            mid_int = int(mid)
                            prev_info = existing_by_id.get(mid_int, {
                                'section': 'Key Facts',
                                'content': ''
                            })
                            memory_manager.update_memory(mid_int, content, config)
                            if prev_info.get('content', '') != content:
                                updated += 1
                                applied_updates.append({
                                    'id': mid_int,
                                    'section': prev_info.get('section', 'Key Facts'),
                                    'before': prev_info.get('content', ''),
                                    'after': content
                                })
                            existing_by_id[mid_int] = {
                                'section': prev_info.get('section', 'Key Facts'),
                                'content': content
                            }
                        except (ValueError, TypeError, memory_manager.MemoryStoreError) as exc:
                            _log_internal_error('run_memory_management update skipped', exc)

                # Process deletions (highest IDs first to avoid index shifting).
                if isinstance(deleted_ids, list):
                    for mid in sorted(deleted_ids, reverse=True):
                        try:
                            mid_int = int(mid)
                            prev_info = existing_by_id.get(mid_int, {
                                'section': 'Key Facts',
                                'content': ''
                            })
                            memory_manager.delete_memory(mid_int, config)
                            deleted += 1
                            applied_deletes.append({
                                'id': mid_int,
                                'section': prev_info.get('section', 'Key Facts'),
                                'content': prev_info.get('content', '')
                            })
                            existing_by_id.pop(mid_int, None)
                        except (ValueError, TypeError, memory_manager.MemoryStoreError) as exc:
                            _log_internal_error('run_memory_management delete skipped', exc)

                # Process additions.
                if isinstance(new_memories, list) and new_memories:
                    memory_manager.add_memories_bulk(new_memories, config)

                pre_counts = _build_counts(existing_flat)
                post_flat = memory_manager.read_flat(config)
                post_counts = _build_counts(post_flat)

                added_counts = {}
                removed_counts = {}

                for key, post_count in post_counts.items():
                    delta = int(post_count or 0) - int(pre_counts.get(key, 0) or 0)
                    if delta > 0:
                        added_counts[key] = delta

                for key, pre_count in pre_counts.items():
                    delta = int(pre_count or 0) - int(post_counts.get(key, 0) or 0)
                    if delta > 0:
                        removed_counts[key] = delta

                # Updates naturally look like remove+add in multiset diff; neutralize them.
                for change in applied_updates:
                    section = str(change.get('section', 'Key Facts') or 'Key Facts').strip() or 'Key Facts'
                    before_key = (section, str(change.get('before', '') or '').strip())
                    after_key = (section, str(change.get('after', '') or '').strip())
                    if before_key[1]:
                        _decrement_count(removed_counts, before_key)
                    if after_key[1]:
                        _decrement_count(added_counts, after_key)

                applied_additions = []
                for key, count in added_counts.items():
                    section, content_value = key
                    for _ in range(int(count or 0)):
                        applied_additions.append({
                            'section': section,
                            'content': content_value
                        })

                added = len(applied_additions)

                parts = []
                if added:
                    parts.append(f"+{added} added")
                if updated:
                    parts.append(f"~{updated} updated")
                if deleted:
                    parts.append(f"-{deleted} removed")
                if parts:
                    emit_chat('console_log', {
                        'message': f"[{datetime.now().strftime('%H:%M:%S')}] Memory managed ({', '.join(parts)})"
                    })
                    emit_chat('memory_updated', {'added': added, 'updated': updated, 'deleted': deleted})

                current_count = len(post_flat)

                compact_lines = []
                for item in applied_additions[:20]:
                    compact_lines.append(
                        f"[ADD] ({item.get('section', 'Key Facts')}) {_preview_text(item.get('content', ''), limit=220)}"
                    )
                for item in applied_updates[:20]:
                    compact_lines.append(
                        f"[EDIT] ID {item.get('id')} ({item.get('section', 'Key Facts')}) "
                        f"{_preview_text(item.get('before', ''), limit=90)} -> {_preview_text(item.get('after', ''), limit=90)}"
                    )
                for item in applied_deletes[:20]:
                    compact_lines.append(
                        f"[REMOVE] ID {item.get('id')} ({item.get('section', 'Key Facts')}) {_preview_text(item.get('content', ''), limit=220)}"
                    )

                overflow = (
                    max(0, len(applied_additions) - 20)
                    + max(0, len(applied_updates) - 20)
                    + max(0, len(applied_deletes) - 20)
                )
                if overflow > 0:
                    compact_lines.append(f"[EDIT] ... and {overflow} more memory changes")

                if not compact_lines:
                    compact_lines.append('[EDIT] No memory changes')

                report_text = '\n'.join(compact_lines)

                if added:
                    first_added = applied_additions[0]
                    emit_memory_thinking(
                        f"Added sample: ({first_added.get('section', 'Key Facts')}) "
                        f"{_preview_text(first_added.get('content', ''), limit=100)}"
                    )
                if updated:
                    first_updated = applied_updates[0]
                    emit_memory_thinking(
                        f"Edited sample ID {first_updated.get('id')}: "
                        f"{_preview_text(first_updated.get('before', ''), limit=60)} -> "
                        f"{_preview_text(first_updated.get('after', ''), limit=60)}"
                    )
                if deleted:
                    first_deleted = applied_deletes[0]
                    emit_memory_thinking(
                        f"Removed sample ID {first_deleted.get('id')}: "
                        f"{_preview_text(first_deleted.get('content', ''), limit=100)}"
                    )

                if parts:
                    emit_memory_thinking(f"Applied memory edits: {', '.join(parts)}. Current total: {current_count}.")
                    finalize_memory_stream(report_text)
                else:
                    emit_memory_thinking(f"No memory edits needed. Current total: {current_count}.")
                    finalize_memory_stream(report_text)
            except memory_manager.MemoryStoreError as exc:
                _log_internal_error('run_memory_management storage error', exc)
                emit_chat('console_log', {
                    'message': f"[{datetime.now().strftime('%H:%M:%S')}] Memory extraction skipped: storage unavailable"
                })
                emit_memory_thinking('Memory storage error encountered while applying edits.')
                finalize_memory_stream('Memory extraction skipped: storage unavailable.', status='error')
        except Exception as e:
            _log_internal_error('run_memory_management unexpected error', e)
            emit_chat('console_log', {
                'message': f"[{datetime.now().strftime('%H:%M:%S')}] Memory extraction skipped due to an internal error"
            })

    def emit_leader_task_distribution_summary(tasks_payload, model_id):
        if not isinstance(tasks_payload, dict):
            return

        def clean_task_value(key):
            return str(tasks_payload.get(key, 'SKIP') or 'SKIP').strip() or 'SKIP'

        analysis_text = str(tasks_payload.get('analysis', '') or '').strip() or 'Routing only'
        researcher_task = clean_task_value('researcher_task')
        creator_task = clean_task_value('creator_task')
        analyzer_task = clean_task_value('analyzer_task')
        verifier_task = clean_task_value('verifier_task')
        direct_response_preview = str(tasks_payload.get('direct_response', '') or '').strip()

        summary_payload = {
            'analysis': analysis_text,
            'researcher_task': researcher_task,
            'creator_task': creator_task,
            'analyzer_task': analyzer_task,
            'verifier_task': verifier_task,
            'direct_response': direct_response_preview
        }
        summary_text = json.dumps(summary_payload, ensure_ascii=False, indent=2)

        message_id = uuid.uuid4().hex
        bot_id = f"council-leader-distribution-{uuid.uuid4().hex[:8]}"
        label = f"Leader - Task Distribution ({model_id})"
        ts = datetime.now().strftime("%H:%M:%S")

        append_conversation_message(
            conversation_id,
            role='assistant',
            content=f"[{label}] {summary_text}",
            raw_markdown=summary_text,
            id=message_id,
            bot_name=label,
            bot_id=bot_id,
            run_group_id=run_group_id,
            run_id=message_id,
            role_name='Leader',
            model_id=model_id,
            thinking='',
            stream_status='done',
            is_final_response=False,
            event_kind='task_distribution'
        )

        emit_chat('ai_response_start', {
            'bot_name': label,
            'bot_id': bot_id,
            'timestamp': ts,
            'message_id': message_id,
            'run_id': message_id,
            'run_group_id': run_group_id,
            'role_name': 'Leader',
            'model_id': model_id,
            'is_final_response': False,
            'event_kind': 'task_distribution'
        })

        emit_chat('ai_response_end', {
            'bot_name': label,
            'bot_id': bot_id,
            'message': summary_text,
            'thinking': '',
            'timestamp': ts,
            'message_id': message_id,
            'run_id': message_id,
            'run_group_id': run_group_id,
            'role_name': 'Leader',
            'model_id': model_id,
            'is_final_response': False,
            'event_kind': 'task_distribution'
        })

        auto_save_chat()

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
            f"[{timestamp}] Council workflow started (mode: {workflow_mode}, "
            f"history: {history_context_mode})"
        )
    })
    leader_skills_context = ''

    # ── Inject cross-chat memory into system prompt (always-on, like ChatGPT/Grok) ──
    memory_context = memory_manager.build_memory_context(config)
    if memory_context:
        user_system_prompt = (user_system_prompt + memory_context) if user_system_prompt else memory_context.strip()
        emit_chat('console_log', {
            'message': f"[{timestamp}] Cross-chat memory injected ({len(memory_manager.read_flat(config))} memories)"
        })

    # FAST mode: Leader-only direct response path.
    if workflow_mode == 'fast':
        for role in ('MarkReader', 'Researcher', 'Creator', 'Analyzer', 'Verifier'):
            emit_chat('council_status', {'role': role, 'status': 'skipped'})
        emit_chat('council_status', {'role': 'Leader', 'status': 'waiting'})

        leader_model_id = config.get('Leader', '')
        leader_info = get_model_info(leader_model_id)
        leader_fast_sys = LEADER_FAST_PROMPT
        if user_system_prompt:
            leader_fast_sys = user_system_prompt + "\n\n" + leader_fast_sys

        emit_chat('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Fast mode active: Leader-only response."
        })

        fast_response = run_council_role(
            role_name='Leader',
            role_label=f'Leader - Final Response ({leader_model_id})',
            model_id=leader_model_id,
            system_prompt=leader_fast_sys,
            user_prompt=user_message,
            chat_history=chat_history,
            conversation_id=conversation_id,
            run_group_id=run_group_id,
            support_images=leader_info.get('support_images', False),
            support_pdf_input=leader_info.get('support_pdf_input', False),
            on_stream_progress=auto_save_chat
        )

        if stop_if_aborted():
            return

        auto_save_chat()
        run_memory_management(council_results_local={}, direct_response_text=fast_response or '')
        complete_workflow(include_run_group_end=True)
        return

    # Set all roles to waiting
    for role in COUNCIL_ROLES:
        emit_chat('council_status', {'role': role, 'status': 'waiting'})

    # ── Step 0: MarkReader selects markdown skill files ──
    md_reader_model_id = config.get('MarkReader', '') or config.get('MD_Reader', '')
    md_reader_config = _normalized_md_reader_config(config)
    inventory_text, inventory_meta = build_md_reader_inventory(config)
    selected_files = []

    if not md_reader_config.get('enabled', True):
        emit_chat('council_status', {'role': 'MarkReader', 'status': 'skipped'})
        emit_chat('console_log', {
            'message': f"[{timestamp}] MarkReader: disabled by config"
        })
    elif inventory_meta.get('status') != 'ok':
        emit_chat('council_status', {'role': 'MarkReader', 'status': 'skipped'})
        emit_chat('console_log', {
            'message': f"[{timestamp}] MarkReader skipped: skills inventory unavailable ({inventory_meta.get('status')})"
        })
    elif not md_reader_model_id:
        emit_chat('council_status', {'role': 'MarkReader', 'status': 'skipped'})
        emit_chat('console_log', {
            'message': f"[{timestamp}] MarkReader skipped: no model configured"
        })
    else:
        md_reader_info = get_model_info(md_reader_model_id)
        md_reader_system = MARK_READER_PROMPT + f"\n\n{inventory_text}"
        if user_system_prompt:
            md_reader_system = user_system_prompt + "\n\n" + md_reader_system

        emit_chat('console_log', {
            'message': f"[{timestamp}] MarkReader ({md_reader_model_id}) selecting markdown files..."
        })

        md_reader_raw = run_council_role(
            role_name='MarkReader',
            role_label=f'MarkReader ({md_reader_model_id})',
            model_id=md_reader_model_id,
            system_prompt=md_reader_system,
            user_prompt=(
                "Select relevant markdown files for Leader context based on the user request."
                f"\n\nUser request:\n{user_message}"
            ),
            chat_history=chat_history,
            conversation_id=conversation_id,
            run_group_id=run_group_id,
            support_images=md_reader_info.get('support_images', False),
            support_pdf_input=md_reader_info.get('support_pdf_input', False),
            on_stream_progress=auto_save_chat
        )

        if stop_if_aborted():
            return

        md_reader_payload = _parse_md_reader_payload(md_reader_raw)
        candidate_files = md_reader_payload.get('selected_files', [])
        selected_files, validate_meta = _validate_selected_skill_files(config, candidate_files)
        rejected = validate_meta.get('rejected_files', [])
        if rejected:
            emit_chat('console_log', {
                'message': f"[{timestamp}] MarkReader rejected invalid files: {', '.join(rejected)}"
            })
        if selected_files:
            emit_chat('console_log', {
                'message': f"[{timestamp}] MarkReader selected files: {', '.join(selected_files)}"
            })
        else:
            emit_chat('console_log', {
                'message': f"[{timestamp}] MarkReader selected no files ({md_reader_payload.get('reason', 'no reason provided')})"
            })

    leader_skills_context, skills_meta = build_leader_skills_context_from_selected(config, selected_files)
    if skills_meta.get('status') == 'loaded':
        joined = ', '.join(skills_meta.get('loaded_files', []))
        emit_chat('console_log', {
            'message': f"[{timestamp}] Leader skill context loaded from MarkReader: {joined}"
        })
    elif skills_meta.get('status') == 'none_selected':
        emit_chat('console_log', {
            'message': f"[{timestamp}] Leader skill context empty: MarkReader selected no markdown"
        })

    # ── Step 1: Leader distributes tasks ──
    leader_model_id = config.get('Leader', '')
    leader_info = get_model_info(leader_model_id)
    leader_sys = LEADER_DISTRIBUTE_PROMPT
    if leader_skills_context:
        leader_sys += leader_skills_context
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
        support_pdf_input=leader_info.get('support_pdf_input', False),
        on_stream_progress=auto_save_chat,
        internal_orchestration=True
    )

    if stop_if_aborted():
        return

    # Parse task distribution with resilient fallback plan.
    if task_distribution_raw is None:
        emit_chat('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Leader failed to return task distribution. Falling back to safe default task plan."
        })
        tasks = _build_safe_default_task_distribution()
    else:
        try:
            tasks = _parse_task_distribution_payload(task_distribution_raw)
        except (json.JSONDecodeError, Exception) as first_error:
            retry_ts = datetime.now().strftime('%H:%M:%S')
            emit_chat('console_log', {
                'message': f"[{retry_ts}] Task distribution JSON parse failed once; retrying with stricter format request."
            })

            strict_retry_system = (
                leader_sys
                + "\n\nSTRICT RETRY INSTRUCTION: Return only one valid JSON object that matches the required schema exactly."
                + " Do not include markdown, code fences, commentary, or extra keys."
                + " If unsure, set non-applicable tasks to SKIP and direct_response to an empty string."
                + "\nRequired keys: analysis, researcher_task, creator_task, analyzer_task, verifier_task, direct_response."
            )

            try:
                retry_raw = completion_response(
                    model=leader_model_id,
                    system_prompt=strict_retry_system,
                    user_prompt=user_message,
                    chat_history=chat_history,
                    temperature=0.2
                )
                tasks = _parse_task_distribution_payload(retry_raw)
            except (json.JSONDecodeError, Exception) as retry_error:
                emit_chat('console_log', {
                    'message': f"[{datetime.now().strftime('%H:%M:%S')}] Failed to parse task distribution after strict retry: {str(retry_error)} (first error: {str(first_error)}). Falling back to safe default task plan."
                })
                tasks = _build_safe_default_task_distribution()

    direct_response = tasks.get('direct_response', '')
    role_task_keys = ('researcher_task', 'creator_task', 'analyzer_task', 'verifier_task')
    all_roles_skipped = all(_is_skip_task(tasks.get(task_key)) for task_key in role_task_keys)

    if workflow_mode == 'heavy' and all_roles_skipped:
        emit_chat('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Heavy mode override: forcing council path instead of direct response."
        })
        tasks['direct_response'] = ''
        tasks['creator_task'] = (
            'Create the primary answer that fully addresses the user request in clear markdown.'
        )
        tasks['verifier_task'] = (
            'Audit the Creator output for factual/logical issues and report only concrete problems.'
        )
        all_roles_skipped = False
        direct_response = ''

    if not all_roles_skipped:
        emit_leader_task_distribution_summary(tasks, leader_model_id)

    if direct_response and all_roles_skipped and workflow_mode != 'heavy':
        direct_ts = datetime.now().strftime("%H:%M:%S")
        emit_chat('console_log', {
            'message': f"[{direct_ts}] Leader direct-response mode activated (all role tasks SKIP)."
        })

        for role_name in ('Researcher', 'Creator', 'Analyzer', 'Verifier'):
            emit_chat('council_status', {
                'role': role_name,
                'status': 'skipped',
                'run_group_id': run_group_id
            })

        final_message_id = uuid.uuid4().hex
        final_bot_id = f"council-leader-final-{uuid.uuid4().hex[:8]}"
        final_label = f"Leader - Final Response ({leader_model_id})"

        append_conversation_message(
            conversation_id,
            role='assistant',
            content=f"[{final_label}] {direct_response}",
            raw_markdown=direct_response,
            id=final_message_id,
            bot_name=final_label,
            bot_id=final_bot_id,
            run_group_id=run_group_id,
            run_id=final_message_id,
            role_name='Leader',
            model_id=leader_model_id,
            thinking='',
            stream_status='done',
            is_final_response=True
        )

        emit_chat('ai_response_start', {
            'bot_name': final_label,
            'bot_id': final_bot_id,
            'timestamp': direct_ts,
            'message_id': final_message_id,
            'run_id': final_message_id,
            'run_group_id': run_group_id,
            'role_name': 'Leader',
            'model_id': leader_model_id,
            'is_final_response': True
        })

        emit_chat('ai_response_end', {
            'bot_name': final_label,
            'bot_id': final_bot_id,
            'message': direct_response,
            'thinking': '',
            'timestamp': direct_ts,
            'message_id': final_message_id,
            'run_id': final_message_id,
            'run_group_id': run_group_id,
            'role_name': 'Leader',
            'model_id': leader_model_id,
            'is_final_response': True
        })

        emit_chat('council_status', {
            'role': 'Leader',
            'status': 'done',
            'run_group_id': run_group_id
        })

        auto_save_chat()
        run_memory_management(council_results_local={}, direct_response_text=direct_response)
        complete_workflow(include_run_group_end=True)
        return

    if direct_response and not all_roles_skipped:
        emit_chat('console_log', {
            'message': f"[{datetime.now().strftime('%H:%M:%S')}] Leader direct_response ignored because one or more role tasks were assigned."
        })

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

    # ── Step 3: Creator ──
    creative_context = ""
    if 'Researcher' in council_results:
        creative_context = f"\n\n===== Researcher's Findings (for your context) =====\n{council_results['Researcher']}"
    _run_optional_council_role(
        role_name='Creator',
        task_key='creator_task',
        base_prompt=CREATOR_PROMPT,
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
    if 'Creator' in council_results:
        verifier_context += f"\n\n===== Creator's Output =====\n{council_results['Creator']}"
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

    # ── Step 5.5: Post-round debate stage (add-on, first round remains unchanged) ──
    _run_post_round_debate_stage(
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
        stop_if_aborted=stop_if_aborted
    )
    if stop_if_aborted():
        return

    # ── Step 6: Leader synthesizes all results ──
    if council_results:
        timestamp = datetime.now().strftime("%H:%M:%S")
        emit_chat('console_log', {
            'message': f"[{timestamp}] Leader ({leader_model_id}) combining results..."
        })

        ordered_keys = ['Researcher', 'Creator', 'Analyzer', 'Verifier', 'Verifier_Debate']
        results_text = ""
        for key in ordered_keys:
            if key in council_results:
                label = key.replace('_', ' ')
                results_text += f"\n\n===== {label} =====\n{council_results[key]}"

        combine_sys = LEADER_COMBINE_PROMPT
        if leader_skills_context:
            combine_sys += leader_skills_context
        combine_sys += f"\n\nTeam outputs:{results_text}"
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
            support_pdf_input=leader_info.get('support_pdf_input', False),
            on_stream_progress=auto_save_chat
        )
        auto_save_chat()

    # ── Step 7: Silent memory management (runs in background, like ChatGPT) ──
    run_memory_management(council_results_local=council_results)

    # Signal all done
    complete_workflow(include_run_group_end=True)

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

# ─── Memory Management API Endpoints ────────────────────────────────

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
