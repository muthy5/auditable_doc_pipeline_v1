from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from .config import PipelineConfig
from .document_classifier import SUPPORTED_DOCUMENT_TYPES
from .pipeline import AuditablePipeline

_OLLAMA_MODEL_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def _parse_ollama_base_url(value: str) -> str:
    """Validate Ollama base URL argument."""
    if not (value.startswith("http://") or value.startswith("https://")):
        raise argparse.ArgumentTypeError("--ollama-base-url must start with http:// or https://")
    if any(ch in value for ch in [" ", ";", "\n", "\r"]):
        raise argparse.ArgumentTypeError("--ollama-base-url must not contain spaces, semicolons, or newline characters")
    return value


def _parse_ollama_model(value: str) -> str:
    """Validate Ollama model argument."""
    if value == "":
        return value
    if not _OLLAMA_MODEL_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("--ollama-model may contain only letters, numbers, hyphens, underscores, colons, and periods")
    return value


def _parse_ollama_max_retries(value: str) -> int:
    """Validate Ollama max retry argument."""
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ollama-max-retries must be an integer") from exc
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("--ollama-max-retries must be 0 or greater")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run the auditable document pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input text document.")
    parser.add_argument("--runs-dir", default="runs", help="Directory where run artifacts will be written.")
    parser.add_argument("--run-dir", default="", help="Existing run directory (used with --resume).")
    parser.add_argument("--backend", default="demo", choices=["demo", "ollama", "claude"], help="Backend to use.")
    parser.add_argument("--doc-id", default="doc_001", help="Document ID.")
    parser.add_argument("--title", default=None, help="Optional document title.")
    parser.add_argument("--goal", default="Identify missing information and organize the document into an actionable structure.")
    parser.add_argument("--document-type", default="auto", choices=["auto", *sorted(SUPPORTED_DOCUMENT_TYPES)], help="Document type to audit against, or auto to classify.")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434", type=_parse_ollama_base_url)
    parser.add_argument("--ollama-model", default="", type=_parse_ollama_model)
    parser.add_argument("--ollama-timeout-s", default=120.0, type=float)
    parser.add_argument("--ollama-temperature", default=0.0, type=float)
    parser.add_argument("--ollama-num-predict", default=2048, type=int)
    parser.add_argument("--ollama-max-retries", default=2, type=_parse_ollama_max_retries)
    parser.add_argument("--claude-api-key", default="")
    parser.add_argument("--claude-model", default="claude-sonnet-4-20250514")
    parser.add_argument("--enable-search", action="store_true", help="Enable Brave web search enrichment.")
    parser.add_argument("--brave-api-key", default="", help="Brave Search API key (or set BRAVE_API_KEY).")
    parser.add_argument("--reference-dir", default=None, help="Directory containing local reference docs for retrieval.")
    parser.add_argument("--strict", action="store_true", help="Halt on first JSON schema validation failure.")
    parser.add_argument("--parallel-chunks", default=None, type=int, help="Number of chunk workers for pass 01 (default: 4 for claude, 1 for demo/ollama).")
    parser.add_argument("--fast", action="store_true", help="Fast mode: larger chunks, parallel extraction, and skip passes 05/06.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing run from first incomplete pass.")
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan and exit.")
    parser.add_argument("--verbose", action="store_true", help="Set log level to DEBUG.")
    parser.add_argument("--quiet", action="store_true", help="Set log level to ERROR.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    if args.quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    execution_plan = [
        {"pass": name, "order": i + 1, "backend": args.backend}
        for i, (name, _, _) in enumerate(AuditablePipeline.PASS_SEQUENCE)
    ]
    if args.dry_run:
        print(json.dumps(execution_plan, indent=2))
        return

    config = PipelineConfig(
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
        ollama_timeout_s=args.ollama_timeout_s,
        ollama_temperature=args.ollama_temperature,
        ollama_num_predict=args.ollama_num_predict,
        ollama_max_retries=args.ollama_max_retries,
        claude_api_key=args.claude_api_key,
        claude_model=args.claude_model,
        enable_search=args.enable_search,
        brave_api_key=args.brave_api_key or os.environ.get("BRAVE_API_KEY", ""),
        reference_dir=args.reference_dir or "",
    )
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name=args.backend, config=config)

    explicit_run_dir = Path(args.run_dir) if args.run_dir else None
    run_dir = pipeline.run(
        input_path=Path(args.input),
        runs_dir=Path(args.runs_dir),
        run_dir=explicit_run_dir,
        resume=args.resume,
        doc_id=args.doc_id,
        title=args.title,
        user_goal=args.goal,
        strict=args.strict,
        document_type=args.document_type,
        parallel_chunks=args.parallel_chunks,
        fast=args.fast,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
