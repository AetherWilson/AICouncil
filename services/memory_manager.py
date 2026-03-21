"""
Cross-chat memory manager.

Reads and writes `skills/memories/memory.md` to provide persistent,
cross-conversation memory — similar to ChatGPT / Grok / Gemini memory.

Memory file structure:
    # Cross-Chat Memory
    ## User Profile
    - [2026-03-20] ...
    ## Preferences
    - [2026-03-20] ...
    ## Key Facts
    - [2026-03-20] ...
    ## Project Context
    - [2026-03-20] ...
"""

import os
import re
import threading
from datetime import datetime

# Known sections in the memory file (in display order).
SECTIONS = ['User Profile', 'Preferences', 'Key Facts', 'Project Context']

DEFAULT_MEMORY_PATH = os.path.join('skills', 'memories', 'memory.md')
DEFAULT_MAX_MEMORIES = 50

_lock = threading.Lock()

# ─── Low-level helpers ───────────────────────────────────────────────

def _memory_path(config=None):
    """Resolve absolute path to the memory markdown file."""
    rel = DEFAULT_MEMORY_PATH
    if config and isinstance(config, dict):
        rel = config.get('memory', {}).get('path', rel) or rel
    return os.path.abspath(rel)


def _ensure_file(path):
    """Create the memory file + parent dirs if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(_blank_template())


def _blank_template():
    lines = [
        '# Cross-Chat Memory',
        '',
        '> This file is automatically managed by the memory system.',
        '> Memories persist across all conversations and are injected into every chat.',
        '> You can also edit this file manually.',
        '',
    ]
    for section in SECTIONS:
        lines.append(f'## {section}')
        lines.append('')
        lines.append('')
    return '\n'.join(lines)


# ─── Parsing ─────────────────────────────────────────────────────────

def _parse_sections(text):
    """Parse memory markdown into {section_name: [entry, ...]}."""
    sections = {s: [] for s in SECTIONS}
    current_section = None
    for line in text.splitlines():
        stripped = line.strip()
        # Detect section header
        m = re.match(r'^##\s+(.+)$', stripped)
        if m:
            header = m.group(1).strip()
            if header in sections:
                current_section = header
            else:
                current_section = None
            continue
        # Collect bullet entries
        if current_section and stripped.startswith('- '):
            sections[current_section].append(stripped[2:].strip())
    return sections


def _serialize_sections(sections):
    """Turn parsed sections back into the markdown file content."""
    lines = [
        '# Cross-Chat Memory',
        '',
        '> This file is automatically managed by the memory system.',
        '> Memories persist across all conversations and are injected into every chat.',
        '> You can also edit this file manually.',
        '',
    ]
    for section in SECTIONS:
        lines.append(f'## {section}')
        lines.append('')
        entries = sections.get(section, [])
        for entry in entries:
            lines.append(f'- {entry}')
        lines.append('')
    return '\n'.join(lines)


def _today():
    return datetime.now().strftime('%Y-%m-%d')


# ─── Public API ──────────────────────────────────────────────────────

def is_enabled(config=None):
    """Check if the memory system is enabled."""
    if config and isinstance(config, dict):
        mem_cfg = config.get('memory', {})
        if isinstance(mem_cfg, dict):
            enabled = mem_cfg.get('enabled', True)
            if isinstance(enabled, bool):
                return enabled
            if isinstance(enabled, str):
                return enabled.strip().lower() in ('1', 'true', 'yes', 'on')
    return True


def max_memories(config=None):
    """Return the configured cap on total memory entries."""
    if config and isinstance(config, dict):
        mem_cfg = config.get('memory', {})
        if isinstance(mem_cfg, dict):
            try:
                val = int(mem_cfg.get('max_memories', DEFAULT_MAX_MEMORIES))
                return max(1, val)
            except (TypeError, ValueError):
                pass
    return DEFAULT_MAX_MEMORIES


def auto_extract_enabled(config=None):
    """Check if auto-extraction is turned on."""
    if config and isinstance(config, dict):
        mem_cfg = config.get('memory', {})
        if isinstance(mem_cfg, dict):
            val = mem_cfg.get('auto_extract', True)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() in ('1', 'true', 'yes', 'on')
    return True


def read_all(config=None):
    """Read memory file and return parsed sections dict."""
    path = _memory_path(config)
    _ensure_file(path)
    with _lock:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    return _parse_sections(text)


def read_flat(config=None):
    """Return all memory entries as a flat list of dicts."""
    sections = read_all(config)
    entries = []
    idx = 0
    for section in SECTIONS:
        for entry in sections.get(section, []):
            entries.append({
                'id': idx,
                'section': section,
                'content': entry
            })
            idx += 1
    return entries


def build_memory_context(config=None):
    """Build the string to inject into system prompts.

    Returns an empty string if disabled or no memories exist.
    """
    if not is_enabled(config):
        return ''

    sections = read_all(config)
    total = sum(len(v) for v in sections.values())
    if total == 0:
        return ''

    lines = [
        '',
        '',
        '===== CROSS-CHAT MEMORY (persistent facts about the user) ====='
    ]
    for section in SECTIONS:
        entries = sections.get(section, [])
        if entries:
            lines.append(f'\n### {section}')
            for entry in entries:
                lines.append(f'- {entry}')
    lines.append('')
    return '\n'.join(lines)


def add_memory(section, content, config=None):
    """Add a single memory entry to a section. Returns the updated flat list."""
    if section not in SECTIONS:
        raise ValueError(f'Invalid section: {section}. Must be one of {SECTIONS}')

    path = _memory_path(config)
    _ensure_file(path)

    content = content.strip()

    cap = max_memories(config)

    with _lock:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        sections = _parse_sections(text)

        # Dedup: skip if identical entry already exists
        if content not in sections[section]:
            sections[section].append(content)

        # Enforce global cap (FIFO across all sections)
        total = sum(len(v) for v in sections.values())
        while total > cap:
            # Remove the oldest entry from the first non-empty section
            for s in SECTIONS:
                if sections[s]:
                    sections[s].pop(0)
                    total -= 1
                    break

        with open(path, 'w', encoding='utf-8') as f:
            f.write(_serialize_sections(sections))

    return read_flat(config)


def add_memories_bulk(entries, config=None):
    """Add multiple memories at once.

    `entries` is a list of dicts: [{"section": "...", "content": "..."}, ...]
    """
    path = _memory_path(config)
    _ensure_file(path)
    cap = max_memories(config)

    with _lock:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        sections = _parse_sections(text)

        for entry in entries:
            section = entry.get('section', 'Key Facts')
            content = entry.get('content', '').strip()
            if not content:
                continue
            if section not in SECTIONS:
                section = 'Key Facts'
            
            if content not in sections[section]:
                sections[section].append(content)

        # Enforce global cap
        total = sum(len(v) for v in sections.values())
        while total > cap:
            for s in SECTIONS:
                if sections[s]:
                    sections[s].pop(0)
                    total -= 1
                    break

        with open(path, 'w', encoding='utf-8') as f:
            f.write(_serialize_sections(sections))


def update_memory(memory_id, new_content, config=None):
    """Update a memory entry by its flat-list index."""
    path = _memory_path(config)
    _ensure_file(path)

    with _lock:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        sections = _parse_sections(text)

        idx = 0
        found = False
        for section in SECTIONS:
            for i, entry in enumerate(sections[section]):
                if idx == memory_id:
                    sections[section][i] = new_content.strip()
                    found = True
                    break
                idx += 1
            if found:
                break

        if not found:
            raise ValueError(f'Memory id {memory_id} not found')

        with open(path, 'w', encoding='utf-8') as f:
            f.write(_serialize_sections(sections))

    return read_flat(config)


def delete_memory(memory_id, config=None):
    """Delete a memory entry by its flat-list index."""
    path = _memory_path(config)
    _ensure_file(path)

    with _lock:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        sections = _parse_sections(text)

        idx = 0
        found = False
        for section in SECTIONS:
            for i, entry in enumerate(sections[section]):
                if idx == memory_id:
                    sections[section].pop(i)
                    found = True
                    break
                idx += 1
            if found:
                break

        if not found:
            raise ValueError(f'Memory id {memory_id} not found')

        with open(path, 'w', encoding='utf-8') as f:
            f.write(_serialize_sections(sections))

    return read_flat(config)


def clear_all(config=None):
    """Wipe all memory entries (keeps the file structure)."""
    path = _memory_path(config)
    with _lock:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(_blank_template())
