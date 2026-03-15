from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import AuditablePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the auditable document pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input text document.")
    parser.add_argument("--runs-dir", default="runs", help="Directory where run artifacts will be written.")
    parser.add_argument("--backend", default="demo", choices=["demo"], help="Backend to use.")
    parser.add_argument("--doc-id", default="doc_001", help="Document ID.")
    parser.add_argument("--title", default=None, help="Optional document title.")
    parser.add_argument(
        "--goal",
        default="Identify missing information and organize the document into an actionable structure.",
        help="User goal passed to normalize_request.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name=args.backend)
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
