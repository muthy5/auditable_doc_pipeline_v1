from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    chunk_target_min_words: int = 2000
    chunk_target_max_words: int = 3000
    chunk_hard_max_words: int = 4000
    chunk_overlap_max_words: int = 120
    encoding: str = "utf-8"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    ollama_timeout_s: float = 120.0
    ollama_temperature: float = 0.0
    ollama_num_predict: int = 2048
    ollama_max_retries: int = 2
    claude_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    claude_tokens_per_minute: int = 80000
    claude_fast_model: str = "claude-haiku-4-5-20251001"
    claude_max_retries: int = 4
    enable_prompt_caching: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    brave_api_key: str = ""
    enable_search: bool = False
    enable_fallback_search: bool = False
    reference_dir: str = ""


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
