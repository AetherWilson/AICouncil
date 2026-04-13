import json
import os
import time
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_COUNCIL_CONFIG = {
    'Leader': 'gpt-4o',
    'history_context_mode': 'final_only',
    'skills': {
        'enabled': True,
        'folder': 'skills',
        'allow_legacy_flat': True,
        'model_map': {
            'researcher-skill': 'gpt-4o-mini',
            'creator-skill': 'gpt-4o',
            'analyzer-skill': 'gpt-4o',
            'verifier-skill': 'gpt-4o'
        }
    },
    'agent_loop': {
        'warning_interval': 10
    },
    'memory': {
        'enabled': True,
        'auto_extract': True,
        'path': os.path.join('skills', 'memories', 'memory.md')
    },
    'document_processing': {
        'pdf_enable_native_input': True,
        'pdf_archive_enabled': True,
        'pdf_intent_budget_enabled': True,
        'pdf_visual_enabled': True,
        'pdf_visual_intent_gating': True,
        'pdf_visual_max_pages': 3,
        'pdf_visual_dpi': 150,
        'pdf_visual_max_total_bytes': 6291456,
        'pdf_visual_max_dimension': 2048,
        'pdf_pdf_binary_intent_gating': True,
        'pdf_retrieval_candidate_multiplier': 3,
        'pdf_retrieval_top_k': 8,
        'pdf_retrieval_max_chars': 12000,
        'pdf_retrieval_cache_enabled': True,
        'pdf_retrieval_cache_ttl_seconds': 600,
        'pdf_expand_on_low_confidence': True,
        'pdf_low_confidence_threshold': 0.28,
        'pdf_page_focus_neighbor_radius': 1,
        'pdf_page_focus_max_chars': 3500,
        'pdf_embedding_enabled': False,
        'pdf_embedding_model': '',
        'pdf_text_max_chars': 20000,
        'pdf_context_max_chars': 12000
    }
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

    def _read_json_cached(self, filename: str, fallback: Any) -> Any:
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
            logger.warning("Invalid JSON in %s: %s", path, exc)
            value = fallback
        except UnicodeDecodeError as exc:
            logger.warning("Invalid encoding in %s: %s", path, exc)
            value = fallback
        except OSError as exc:
            logger.warning("Unable to read %s: %s", path, exc)
            value = fallback

        self._cache[path] = {
            'loaded_at': now,
            'value': value
        }
        return value

    def load_config(self) -> Dict[str, Any]:
        config = self._read_json_cached('config.json', dict(DEFAULT_COUNCIL_CONFIG))
        if not isinstance(config, dict):
            return dict(DEFAULT_COUNCIL_CONFIG)
        merged = _deep_merge(DEFAULT_COUNCIL_CONFIG, config)
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
