from __future__ import annotations

from pathlib import Path

from .exceptions import PipelineError


def load_prompt(prompts_dir: Path, filename: str) -> str:
    """Load a prompt template from disk.

    Args:
        prompts_dir: Directory containing prompt files.
        filename: Prompt filename.

    Returns:
        Prompt text.

    Raises:
        PipelineError: If prompt file does not exist.
    """
    path = prompts_dir / filename
    if not path.exists():
        raise PipelineError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
