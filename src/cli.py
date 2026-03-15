from __future__ import annotations

import argparse
import re
from pathlib import Path

from .config import PipelineConfig
from .pipeline import AuditablePipeline


_OLLAMA_MODEL_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def _parse_ollama_base_url(value: str) -> str:
    if not (value.startswith("http://") or value.startswith("https://")):
        raise argparse.ArgumentTypeError("--ollama-base-url must start with http:// or https://")
    if any(ch in value for ch in [" ", ";", "\n", "\r"]):
        raise argparse.ArgumentTypeError(
            "--ollama-base-url must not contain spaces, semicolons, or newline characters"
        )
    return value


def _parse_ollama_model(value: str) -> str:
    if value == "":
        return value
    if not _OLLAMA_MODEL_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "--ollama-model may contain only letters, numbers, hyphens, underscores, colons, and periods"
        )
    return value


def _parse_ollama_max_retries(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ollama-max-retries must be an integer") from exc

    if parsed_value < 0:
        raise argparse.ArgumentTypeError("--ollama-max-retries must be 0 or greater")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the auditable document pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input text document.")
    parser.add_argument("--runs-dir", default="runs", help="Directory where run artifacts will be written.")
    parser.add_argument("--backend", default="demo", choices=["demo", "ollama"], help="Backend to use.")
    parser.add_argument("--doc-id", default="doc_001", help="Document ID.")
    parser.add_argument("--title", default=None, help="Optional document title.")
    parser.add_argument(
        "--goal",
        default="Identify missing information and organize the document into an actionable structure.",
        help="User goal passed to normalize_request.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
        type=_parse_ollama_base_url,
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--ollama-model",
        default="",
        type=_parse_ollama_model,
        help="Ollama model name (required when --backend ollama).",
    )
    parser.add_argument("--ollama-timeout-s", default=120.0, type=float, help="Ollama request timeout in seconds.")
    parser.add_argument("--ollama-temperature", default=0.0, type=float, help="Sampling temperature for Ollama generation.")
    parser.add_argument("--ollama-num-predict", default=2048, type=int, help="Maximum tokens to predict per Ollama call.")
    parser.add_argument(
        "--ollama-max-retries",
        default=2,
        type=_parse_ollama_max_retries,
        help="Retries for invalid/non-JSON Ollama output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
        ollama_timeout_s=args.ollama_timeout_s,
        ollama_temperature=args.ollama_temperature,
        ollama_num_predict=args.ollama_num_predict,
        ollama_max_retries=args.ollama_max_retries,
    )

    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name=args.backend, config=config)
    run_dir = pipeline.run(
        input_path=Path(args.input),
        runs_dir=Path(args.runs_dir),
        doc_id=args.doc_id,
        title=args.title,
        user_goal=args.goal,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
