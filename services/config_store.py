import json
import os
import time
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_COUNCIL_CONFIG = {
    'history_context_mode': 'final_only',
    'council': {
        'chair': 'chair',
        'members': [
            {'id': 'chair', 'name': 'Chair', 'model': 'gpt-4o', 'enabled': True},
            {'id': 'member-1', 'name': 'Member 1', 'model': 'gpt-4o-mini', 'enabled': True},
            {'id': 'member-2', 'name': 'Member 2', 'model': 'gpt-4o', 'enabled': True},
        ],
        'discussion_rounds': 2,
        'max_revisions': 1,
        'turn_timeout_seconds': 120,
    },
    'memory': {
        'enabled': True,
        'auto_extract': False,
        'path': os.path.join('skills', 'memories', 'memory.md'),
    },
}


def _bounded_int(value, default, minimum, maximum):
    try:
        return max(minimum, min(maximum, int(value)))
    except (ValueError, TypeError, OverflowError):
        return default


def normalize_council(config):
    """Validate explicit seats, preserving list order; adapt old model assignments."""
    raw = config.get('council')
    if raw is None:
        defaults = DEFAULT_COUNCIL_CONFIG['council']
        leader = str(config.get('Leader') or defaults['members'][0]['model']).strip()
        models = []
        skills = config.get('skills') if isinstance(config.get('skills'), dict) else {}
        mapping = skills.get('model_map') if isinstance(skills.get('model_map'), dict) else {}
        for role in ('Researcher', 'Creator', 'Creative_Writer', 'Analyzer', 'Verifier'):
            model = str(config.get(role) or '').strip()
            if model and model not in models:
                models.append(model)
        for key in ('researcher-skill', 'creator-skill', 'analyzer-skill', 'verifier-skill'):
            model = str(mapping.get(key) or '').strip()
            if model and model not in models:
                models.append(model)
        if not models:
            models = [leader]
        raw = dict(defaults, members=[{'id': 'chair', 'name': 'Chair', 'model': leader}]
                   + [{'id': f'member-{i}', 'name': f'Member {i}', 'model': model}
                      for i, model in enumerate(models, 1)])
    if not isinstance(raw, dict):
        raise ValueError('council must be an object')
    seats = raw.get('members')
    if not isinstance(seats, list):
        raise ValueError('council.members must be an ordered list')
    members, seen = [], set()
    for seat in seats:
        if not isinstance(seat, dict):
            raise ValueError('Each council member must be an object')
        if seat.get('enabled', True) is False:
            continue
        member_id = str(seat.get('id') or '').strip()
        model = str(seat.get('model') or '').strip()
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', member_id) or member_id in seen:
            raise ValueError('Council member IDs must be unique letters/digits/underscores/hyphens, starting with a letter')
        if not model:
            raise ValueError(f'Missing model for council member {member_id}')
        seen.add(member_id)
        members.append({'id': member_id, 'name': str(seat.get('name') or member_id).strip(),
                        'model': model, 'enabled': True})
    chair = str(raw.get('chair') or 'chair').strip()
    if chair not in seen or len(members) < 2:
        raise ValueError('Council needs an enabled chair and at least one other enabled member')
    return {
        'members': members, 'chair': chair,
        'discussion_rounds': _bounded_int(raw.get('discussion_rounds'), 2, 1, 8),
        'max_revisions': _bounded_int(raw.get('max_revisions'), 1, 0, 5),
        'turn_timeout_seconds': _bounded_int(raw.get('turn_timeout_seconds'), 120, 1, 600),
    }


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    return override if override is not None else base


class ConfigStore:
    """Small cached config/model loader to reduce repeated disk I/O."""

    def __init__(self, base_dir: str, cache_ttl_seconds: float = 2.0) -> None:
        self.base_dir = base_dir
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _read_json_cached(self, filename: str, fallback: Any, strict: bool = False) -> Any:
        path = os.path.join(self.base_dir, filename)
        now = time.time()
        entry = self._cache.get(path)

        if entry and (now - entry['loaded_at']) < self.cache_ttl_seconds:
            return entry['value']

        try:
            # Use utf-8-sig to tolerate BOM-prefixed JSON files produced by some editors/tools.
            with open(path, 'r', encoding='utf-8-sig') as f:
                value = json.load(f)
        except FileNotFoundError:
            value = fallback
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(f'Invalid JSON in {filename}') from exc
            logger.warning("Invalid JSON in %s: %s", path, exc)
            value = fallback
        except UnicodeDecodeError as exc:
            if strict:
                raise ValueError(f'Invalid encoding in {filename}') from exc
            logger.warning("Invalid encoding in %s: %s", path, exc)
            value = fallback
        except OSError as exc:
            if strict:
                raise ValueError(f'Unable to read {filename}') from exc
            logger.warning("Unable to read %s: %s", path, exc)
            value = fallback

        self._cache[path] = {
            'loaded_at': now,
            'value': value
        }
        return value

    def load_config(self) -> Dict[str, Any]:
        config = self._read_json_cached('config.json', {}, strict=True)
        if not isinstance(config, dict):
            raise ValueError('config.json must contain an object')
        merged = _deep_merge(DEFAULT_COUNCIL_CONFIG, config)
        # Normalize the original council object so legacy config and missing seats
        # are not accidentally hidden by deep-merged defaults.
        merged['council'] = normalize_council(config) if config else normalize_council(DEFAULT_COUNCIL_CONFIG)
        if not isinstance(merged.get('memory'), dict):
            raise ValueError('memory must be an object')
        merged['memory']['auto_extract'] = False
        for key in ('Leader', 'lite_model', 'skills', 'agent_loop', 'MarkReader', 'MD_Reader',
                    'Researcher', 'Creator', 'Creative_Writer', 'Analyzer', 'Verifier'):
            merged.pop(key, None)
        return merged

    def load_models(self) -> List[Dict[str, Any]]:
        models = self._read_json_cached('model.json', [])
        if isinstance(models, list):
            return models
        return []


def infer_model_support_images(model_id: str) -> bool:
    """Best-effort vision capability detection when model metadata is unavailable."""
    model_lower = (model_id or '').lower()
    if not model_lower:
        return False

    vision_hints = [
        'gpt-4o', 'gpt-4.1', 'gpt-5', 'o1', 'o3',
        'gemini', 'grok',
        'claude-3', 'claude-opus-4', 'claude-sonnet-4',
        'vision', 'vl', 'pixtral', 'llava', 'qwen-vl'
    ]
    return any(hint in model_lower for hint in vision_hints)


def infer_model_support_pdf_input(model_id: str) -> bool:
    """Best-effort PDF-input capability detection when model metadata is unavailable."""
    model_lower = (model_id or '').lower()
    if not model_lower:
        return False

    pdf_hints = [
        'claude', 'gemini', 'grok', 'deepseek', 'qwen', 'kimi', 'glm',
        'gpt-4o', 'gpt-4.1', 'gpt-5', 'o1', 'o3'
    ]
    return any(hint in model_lower for hint in pdf_hints)


def _normalize_model_alias(model_id: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(model_id or '').strip().lower())


def _resolve_model_alias(models: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    target = str(model_id or '').strip()
    if not target:
        return {}

    for model in models:
        if str(model.get('id') or '').strip() == target:
            return model

    target_norm = _normalize_model_alias(target)
    if not target_norm:
        return {}

    candidates = []
    for model in models:
        candidate_id = str(model.get('id') or '').strip()
        candidate_norm = _normalize_model_alias(candidate_id)
        if not candidate_norm:
            continue

        if candidate_norm == target_norm:
            return model

        if candidate_norm.startswith(target_norm) or target_norm.startswith(candidate_norm):
            delta = abs(len(candidate_norm) - len(target_norm))
            candidates.append((delta, len(candidate_norm), model))

    if not candidates:
        return {}

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def get_model_info(models: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    resolved = _resolve_model_alias(models, model_id)
    if resolved:
        normalized = dict(resolved)
        normalized['id'] = str(model_id or normalized.get('id') or '')
        if 'support_images' not in normalized:
            normalized['support_images'] = infer_model_support_images(model_id)
        if 'support_pdf_input' not in normalized:
            normalized['support_pdf_input'] = infer_model_support_pdf_input(model_id)
        return normalized

    return {
        'id': model_id,
        'name': model_id,
        'support_images': infer_model_support_images(model_id),
        'support_pdf_input': infer_model_support_pdf_input(model_id)
    }
