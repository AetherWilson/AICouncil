import json
import os
import time
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_COUNCIL_CONFIG = {
    'MarkReader': 'gpt-4o',
    'Leader': 'gpt-4o',
    'Researcher': 'gpt-4o',
    'Creator': 'gpt-4o',
    'Analyzer': 'gpt-4o',
    'Verifier': 'gpt-4o',
    'MemWriter': 'gpt-5'
}

LEGACY_ROLE_KEY_MAP = {
    'MD_Reader': 'MarkReader',
    'Creative_Writer': 'Creator'
}


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

    def load_config(self) -> Dict[str, str]:
        config = self._read_json_cached('config.json', dict(DEFAULT_COUNCIL_CONFIG))
        if not isinstance(config, dict):
            return dict(DEFAULT_COUNCIL_CONFIG)
        merged = dict(DEFAULT_COUNCIL_CONFIG)
        for legacy_key, modern_key in LEGACY_ROLE_KEY_MAP.items():
            # Keep old config files working after role key rename.
            if modern_key not in config and legacy_key in config:
                config[modern_key] = config[legacy_key]
        merged.update(config)
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


def get_model_info(models: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    for model in models:
        if model.get('id') == model_id:
            normalized = dict(model)
            if 'support_images' not in normalized:
                normalized['support_images'] = infer_model_support_images(model_id)
            if 'support_pdf_input' not in normalized:
                normalized['support_pdf_input'] = False
            return normalized

    return {
        'id': model_id,
        'name': model_id,
        'support_images': infer_model_support_images(model_id),
        'support_pdf_input': False
    }
