from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    chunk_target_min_words: int = 900
    chunk_target_max_words: int = 1200
    chunk_hard_max_words: int = 1500
    chunk_overlap_max_words: int = 120
    encoding: str = "utf-8"


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    schemas_dir: Path
    prompts_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "RepoPaths":
        return cls(
            root=root,
            schemas_dir=root / "schemas",
            prompts_dir=root / "prompts",
        )
