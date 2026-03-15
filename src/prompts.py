from __future__ import annotations

from pathlib import Path


def load_prompt(prompts_dir: Path, filename: str) -> str:
    path = prompts_dir / filename
    return path.read_text(encoding="utf-8")
