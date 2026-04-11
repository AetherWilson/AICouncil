import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SkillDefinition:
    skill_id: str
    name: str
    description: str
    path: str
    content: str
    frontmatter: Dict[str, object]
    legacy: bool = False


def _normalize_skill_id(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9-]+", "-", str(value or "").strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unnamed-skill"


def _coerce_scalar(value: str):
    text = str(value or "").strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on", "1"}:
        return True
    if lowered in {"false", "no", "off", "0"}:
        return False
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _parse_frontmatter(raw_text: str) -> Tuple[Dict[str, object], str]:
    text = str(raw_text or "")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text.strip()

    lines = text.splitlines()
    if len(lines) < 3:
        return {}, text.strip()

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, text.strip()

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:]).strip()

    frontmatter: Dict[str, object] = {}
    current_list_key: Optional[str] = None

    for raw in fm_lines:
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue

        if current_list_key and line.lstrip().startswith("- "):
            current_value = frontmatter.setdefault(current_list_key, [])
            if isinstance(current_value, list):
                current_value.append(_coerce_scalar(line.lstrip()[2:]))
            continue

        current_list_key = None
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if not value:
            frontmatter[key] = []
            current_list_key = key
            continue

        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                frontmatter[key] = []
                continue
            parts = [part.strip() for part in inner.split(",")]
            frontmatter[key] = [_coerce_scalar(part.strip("\"'")) for part in parts if part]
            continue

        frontmatter[key] = _coerce_scalar(value.strip("\"'"))

    return frontmatter, body


def _load_skill_file(skill_path: str, legacy: bool = False) -> Optional[SkillDefinition]:
    try:
        with open(skill_path, "r", encoding="utf-8-sig") as handle:
            raw = handle.read()
    except OSError:
        return None

    fm, body = _parse_frontmatter(raw)
    base_name = os.path.basename(os.path.dirname(skill_path)) if not legacy else os.path.splitext(os.path.basename(skill_path))[0]

    skill_id = _normalize_skill_id(str(fm.get("name") or base_name))
    name = str(fm.get("name") or base_name)
    description = str(fm.get("description") or "")

    if not description:
        first_line = ""
        for line in body.splitlines():
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
        description = first_line[:240]

    if not body.strip():
        return None

    return SkillDefinition(
        skill_id=skill_id,
        name=name,
        description=description,
        path=skill_path,
        content=body,
        frontmatter=fm,
        legacy=legacy,
    )


def discover_skills(skills_root: str, allow_legacy_flat: bool = True) -> List[SkillDefinition]:
    root = os.path.abspath(skills_root)
    if not os.path.isdir(root):
        return []

    discovered: Dict[str, SkillDefinition] = {}

    # Canonical format: skills/<skill-id>/SKILL.md
    for child in sorted(os.listdir(root)):
        child_path = os.path.join(root, child)
        if not os.path.isdir(child_path):
            continue
        skill_path = os.path.join(child_path, "SKILL.md")
        if not os.path.isfile(skill_path):
            continue
        loaded = _load_skill_file(skill_path, legacy=False)
        if not loaded:
            continue
        discovered[loaded.skill_id] = loaded

    # Fallback read for migration period: flat markdown files in skills/
    if allow_legacy_flat:
        for child in sorted(os.listdir(root)):
            flat_path = os.path.join(root, child)
            if not os.path.isfile(flat_path):
                continue
            if not child.lower().endswith(".md"):
                continue
            if child.lower() == "readme.md":
                continue
            loaded = _load_skill_file(flat_path, legacy=True)
            if not loaded:
                continue
            discovered.setdefault(loaded.skill_id, loaded)

    return sorted(discovered.values(), key=lambda item: item.skill_id)


def build_skill_catalog(skills: List[SkillDefinition]) -> List[Dict[str, object]]:
    catalog = []
    for skill in skills:
        catalog.append({
            "id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "legacy": skill.legacy,
            "path": skill.path,
            "model": skill.frontmatter.get("model", ""),
        })
    return catalog


def get_skill_by_id(skills: List[SkillDefinition], skill_id: str) -> Optional[SkillDefinition]:
    target = _normalize_skill_id(skill_id)
    for skill in skills:
        if skill.skill_id == target:
            return skill
    return None
