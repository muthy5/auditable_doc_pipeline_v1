from __future__ import annotations

from pathlib import Path


def load_prompt(prompts_dir: Path, filename: str) -> str:
    """Load a prompt template from disk.

    Args:
        prompts_dir: Root prompt directory.
        filename: Prompt filename relative to ``prompts_dir``.

    Returns:
        Prompt text content.
    """
    path = prompts_dir / filename
    return path.read_text(encoding="utf-8")
