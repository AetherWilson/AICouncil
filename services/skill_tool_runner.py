import json
import os
import re
import subprocess
import sys

DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_MAX_OUTPUT_CHARS = 12000
MAX_ARGS = 20
MAX_ARG_CHARS = 400
SCRIPT_NAME_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+\.py$')


def _truncate_text(text, max_chars):
    rendered = str(text or '')
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars] + '\n[... output truncated ...]'


def _normalize_script_name(value):
    script_name = str(value or '').strip().replace('\\', '/')
    if '/' in script_name:
        return ''
    if not SCRIPT_NAME_PATTERN.fullmatch(script_name):
        return ''
    return script_name


def _normalize_args(raw_args):
    normalized = []
    if isinstance(raw_args, list):
        for item in raw_args[:MAX_ARGS]:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text[:MAX_ARG_CHARS])
        return normalized

    if isinstance(raw_args, str):
        text = raw_args.strip()
        if text:
            normalized.append(text[:MAX_ARG_CHARS])
    return normalized


def run_skill_script(
    *,
    skill_file_path,
    script_name,
    args=None,
    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
):
    normalized_script = _normalize_script_name(script_name)
    if not normalized_script:
        return {
            'ok': False,
            'error': 'Invalid script name. Expected a single .py filename.',
            'script': str(script_name or ''),
        }

    skill_dir = os.path.abspath(os.path.dirname(str(skill_file_path or '')))
    scripts_dir = os.path.abspath(os.path.join(skill_dir, 'scripts'))

    if not os.path.isdir(scripts_dir):
        return {
            'ok': False,
            'error': 'Skill scripts folder not found.',
            'script': normalized_script,
            'scripts_dir': scripts_dir,
        }

    script_path = os.path.abspath(os.path.join(scripts_dir, normalized_script))
    if os.path.commonpath([scripts_dir, script_path]) != scripts_dir:
        return {
            'ok': False,
            'error': 'Resolved script path is outside the skill scripts folder.',
            'script': normalized_script,
        }

    if not os.path.isfile(script_path):
        return {
            'ok': False,
            'error': f'Script file not found: {normalized_script}',
            'script': normalized_script,
            'scripts_dir': scripts_dir,
        }

    normalized_args = _normalize_args(args)
    command = [sys.executable, script_path] + normalized_args

    try:
        completed = subprocess.run(
            command,
            cwd=skill_dir,
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            'ok': False,
            'error': f'Script timed out after {int(timeout_seconds)} seconds.',
            'script': normalized_script,
            'args': normalized_args,
            'command': command,
        }
    except OSError as exc:
        return {
            'ok': False,
            'error': f'Failed to execute script: {exc}',
            'script': normalized_script,
            'args': normalized_args,
            'command': command,
        }

    stdout_text = _truncate_text(completed.stdout, max_output_chars)
    stderr_text = _truncate_text(completed.stderr, max_output_chars)

    if completed.returncode != 0:
        return {
            'ok': False,
            'error': 'Script returned a non-zero exit code.',
            'script': normalized_script,
            'args': normalized_args,
            'command': command,
            'exit_code': int(completed.returncode),
            'stdout': stdout_text,
            'stderr': stderr_text,
        }

    try:
        parsed_output = json.loads(stdout_text.strip() or '{}')
    except Exception:
        return {
            'ok': False,
            'error': 'Script stdout is not valid JSON.',
            'script': normalized_script,
            'args': normalized_args,
            'command': command,
            'exit_code': int(completed.returncode),
            'stdout': stdout_text,
            'stderr': stderr_text,
        }

    return {
        'ok': True,
        'script': normalized_script,
        'args': normalized_args,
        'command': command,
        'exit_code': int(completed.returncode),
        'result': parsed_output,
        'stdout': stdout_text,
        'stderr': stderr_text,
    }
