from __future__ import annotations

from pathlib import Path

from .exceptions import PipelineError

_prompt_cache: dict[tuple[str, str], str] = {}


def load_prompt(prompts_dir: Path, filename: str) -> str:
    """Load a prompt template from disk (cached after first read).

    Args:
        prompts_dir: Directory containing prompt files.
        filename: Prompt filename.

    Returns:
        Prompt text.

    Raises:
        PipelineError: If prompt file does not exist.
    """
    cache_key = (str(prompts_dir), filename)
    cached = _prompt_cache.get(cache_key)
    if cached is not None:
        return cached
    path = prompts_dir / filename
    if not path.exists():
        raise PipelineError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    _prompt_cache[cache_key] = text
    return text
