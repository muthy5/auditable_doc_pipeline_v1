from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration values for the pipeline."""

    chunk_target_min_words: int = 900
    chunk_target_max_words: int = 1200
    chunk_hard_max_words: int = 1500
    chunk_overlap_max_words: int = 120
    encoding: str = "utf-8"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    ollama_timeout_s: float = 120.0
    ollama_temperature: float = 0.0
    ollama_num_predict: int = 2048
    ollama_max_retries: int = 2


@dataclass(frozen=True)
class RepoPaths:
    """Resolved repository paths used by the pipeline."""

    root: Path
    schemas_dir: Path
    prompts_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> RepoPaths:
        """Create a path bundle from a repository root.

        Args:
            root: Repository root path.

        Returns:
            A ``RepoPaths`` instance with schema and prompt directories resolved.
        """
        return cls(
            root=root,
            schemas_dir=root / "schemas",
            prompts_dir=root / "prompts",
        )
