from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .chunker import chunk_document
from .config import PipelineConfig, RepoPaths
from .exceptions import BackendError, ChunkingError
from .llm_interface import RuleBasedDemoBackend
from .markdown_writer import render_final_answer_markdown
from .merge_engine import merge_chunk_extractions
from .ollama_backend import OllamaBackendConfig, OllamaLocalBackend
from .pass_runner import PassRunner
from .schemas import load_schema
from .validators import validate_chunks, validate_final_output

LOGGER = logging.getLogger(__name__)


def utc_run_id() -> str:
    """Create a UTC timestamp-based run identifier.

    Returns:
        Run identifier in ``YYYYMMDDTHHMMSSZ`` format.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class AuditablePipeline:
    """Coordinates all passes for an auditable document-analysis run."""

    def __init__(self, repo_root: Path, backend_name: str = "demo", config: PipelineConfig | None = None) -> None:
        """Initialize a pipeline instance.

        Args:
            repo_root: Repository root path.
            backend_name: Backend selector (``demo`` or ``ollama``).
            config: Optional runtime config.

        Raises:
            BackendError: If backend selection/configuration is invalid.
        """
        self.repo_paths = RepoPaths.from_root(repo_root)
        self.config = config or PipelineConfig()
        if backend_name == "demo":
            self.backend = RuleBasedDemoBackend()
        elif backend_name == "ollama":
            ollama_config = OllamaBackendConfig(
                base_url=self.config.ollama_base_url,
                model=self.config.ollama_model,
                timeout_s=self.config.ollama_timeout_s,
                temperature=self.config.ollama_temperature,
                num_predict=self.config.ollama_num_predict,
                max_retries=self.config.ollama_max_retries,
            )
            self.backend = OllamaLocalBackend(config=ollama_config)
        else:
            raise BackendError(f"Unsupported backend: {backend_name}")
        self.pass_runner = PassRunner(
            backend=self.backend,
            prompts_dir=self.repo_paths.prompts_dir,
            schemas_dir=self.repo_paths.schemas_dir,
        )

    def describe_execution_plan(self, backend_name: str) -> List[str]:
        """Describe pass execution order for dry-run mode.

        Args:
            backend_name: Backend name shown in the printed plan.

        Returns:
            Human-readable execution-plan lines.
        """
        passes = [
            "00_normalize_request",
            "01_extract_chunk (per chunk)",
            "02_merge_global",
            "03_schema_audit",
            "04_dependency_audit",
            "05_assumption_audit",
            "06_evidence_audit",
            "07_synthesize",
            "08_validate_final",
        ]
        return [f"backend={backend_name}"] + [f"{idx + 1}. {name}" for idx, name in enumerate(passes)]

    def run(
        self,
        input_path: Path,
        runs_dir: Path,
        doc_id: str = "doc_001",
        title: str | None = None,
        user_goal: str = "Identify missing information and organize the document into an actionable structure.",
        requested_deliverable: str = "structured_gap_analysis_and_plan",
    ) -> Path:
        """Execute the full pipeline and write run artifacts.

        Args:
            input_path: Plain-text input file.
            runs_dir: Parent directory for run outputs.
            doc_id: Document identifier.
            title: Optional document title.
            user_goal: High-level user objective.
            requested_deliverable: Requested output type.

        Returns:
            Path to the created run directory.

        Raises:
            ChunkingError: If chunk validation fails.
        """
        run_id = utc_run_id()
        run_dir = runs_dir / run_id
        input_dir = run_dir / "input"
        passes_dir = run_dir / "passes"
        final_dir = run_dir / "final"
        logs_dir = run_dir / "logs"
        for out_dir in [input_dir, passes_dir, final_dir, logs_dir]:
            out_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Starting pipeline run %s", run_id)
        text = input_path.read_text(encoding=self.config.encoding)
        document: Dict[str, Any] = {
            "doc_id": doc_id,
            "title": title,
            "source_path": str(input_path),
            "content_type": "text/plain",
            "encoding": self.config.encoding,
            "text": text,
            "metadata": {"author": None, "created_at": None, "user_goal": user_goal},
        }

        self.pass_runner.validate_with_schema("document.schema.json", document)
        (input_dir / "document.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

        chunks = chunk_document(
            doc=document,
            target_min_words=self.config.chunk_target_min_words,
            target_max_words=self.config.chunk_target_max_words,
            hard_max_words=self.config.chunk_hard_max_words,
            overlap_max_words=self.config.chunk_overlap_max_words,
        )
        chunk_errors = validate_chunks(chunks)
        if chunk_errors:
            raise ChunkingError(f"Chunk validation failed: {chunk_errors}")
        (input_dir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

        normalize = self.pass_runner.run_model_pass(
            pass_name="00_normalize_request",
            prompt_filename="00_normalize_request.txt",
            schema_filename="00_normalize_request.schema.json",
            input_payload={
                "doc_manifest": {"doc_id": doc_id, "title": title},
                "user_goal": user_goal,
                "requested_deliverable": requested_deliverable,
                "user_constraints": ["Do not invent facts", "Surface uncertainty explicitly"],
            },
            output_path=passes_dir / "00_normalize_request.json",
        )

        extraction_dir = passes_dir / "01_extract_chunk"
        chunk_extractions = [
            self.pass_runner.run_model_pass(
                pass_name="01_extract_chunk",
                prompt_filename="01_extract_chunk.txt",
                schema_filename="01_extract_chunk.schema.json",
                input_payload={"task": normalize, "chunk": chunk},
                output_path=extraction_dir / f"{chunk['chunk_id']}.json",
            )
            for chunk in chunks
        ]

        merge = merge_chunk_extractions(doc_id=doc_id, chunk_extractions=chunk_extractions)
        self.pass_runner.write_validated_json("02_merge_global.schema.json", merge, passes_dir / "02_merge_global.json")
        chunk_summaries = [{"chunk_id": i["chunk_id"], "section_role": i["section_role"]} for i in chunk_extractions]

        schema_audit = self.pass_runner.run_model_pass(
            "03_schema_audit",
            "03_schema_audit.txt",
            "03_schema_audit.schema.json",
            {"task": normalize, "merge": merge, "chunk_summaries": chunk_summaries},
            passes_dir / "03_schema_audit.json",
        )
        dependency_audit = self.pass_runner.run_model_pass(
            "04_dependency_audit",
            "04_dependency_audit.txt",
            "04_dependency_audit.schema.json",
            {"task": normalize, "merge": merge, "schema_audit": schema_audit},
            passes_dir / "04_dependency_audit.json",
        )
        assumption_audit = self.pass_runner.run_model_pass(
            "05_assumption_audit",
            "05_assumption_audit.txt",
            "05_assumption_audit.schema.json",
            {"task": normalize, "merge": merge, "schema_audit": schema_audit, "dependency_audit": dependency_audit},
            passes_dir / "05_assumption_audit.json",
        )
        evidence_audit = self.pass_runner.run_model_pass(
            "06_evidence_audit",
            "06_evidence_audit.txt",
            "06_evidence_audit.schema.json",
            {
                "merge": merge,
                "schema_audit": schema_audit,
                "dependency_audit": dependency_audit,
                "assumption_audit": assumption_audit,
            },
            passes_dir / "06_evidence_audit.json",
        )
        synthesis = self.pass_runner.run_model_pass(
            "07_synthesize",
            "07_synthesize.txt",
            "07_synthesize.schema.json",
            {
                "task": normalize,
                "merge": merge,
                "schema_audit": schema_audit,
                "dependency_audit": dependency_audit,
                "assumption_audit": assumption_audit,
                "evidence_audit": evidence_audit,
            },
            passes_dir / "07_synthesize.json",
        )

        validation = validate_final_output(
            synthesis=synthesis,
            task=normalize,
            schema_audit=schema_audit,
            dependency_audit=dependency_audit,
            assumption_audit=assumption_audit,
            evidence_audit=evidence_audit,
            synthesis_schema=load_schema(self.repo_paths.schemas_dir, "07_synthesize.schema.json"),
        )
        self.pass_runner.write_validated_json("08_validate_final.schema.json", validation, passes_dir / "08_validate_final.json")

        (final_dir / "final_answer.json").write_text(json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8")
        (final_dir / "final_answer.md").write_text(render_final_answer_markdown(synthesis), encoding="utf-8")
        (logs_dir / "run.log").write_text(
            "\n".join(
                [
                    f"input={input_path}",
                    f"run_id={run_id}",
                    f"validation_pass={validation['pass']}",
                    f"errors={len(validation['errors'])}",
                    f"warnings={len(validation['warnings'])}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return run_dir
