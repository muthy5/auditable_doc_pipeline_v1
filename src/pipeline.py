from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .chunker import chunk_document
from .config import PipelineConfig, RepoPaths
from .llm_interface import RuleBasedDemoBackend
from .ollama_backend import OllamaBackendConfig, OllamaLocalBackend
from .markdown_writer import render_final_answer_markdown
from .merge_engine import merge_chunk_extractions
from .pass_runner import PassRunner
from .schemas import load_schema
from .validators import validate_chunks, validate_final_output


def utc_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


class AuditablePipeline:
    def __init__(self, repo_root: Path, backend_name: str = "demo", config: PipelineConfig | None = None) -> None:
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
            raise ValueError(f"Unsupported backend: {backend_name}")
        self.pass_runner = PassRunner(
            backend=self.backend,
            prompts_dir=self.repo_paths.prompts_dir,
            schemas_dir=self.repo_paths.schemas_dir,
        )

    def run(
        self,
        input_path: Path,
        runs_dir: Path,
        doc_id: str = "doc_001",
        title: str | None = None,
        user_goal: str = "Identify missing information and organize the document into an actionable structure.",
        requested_deliverable: str = "structured_gap_analysis_and_plan",
    ) -> Path:
        run_id = utc_run_id()
        run_dir = runs_dir / run_id
        input_dir = run_dir / "input"
        passes_dir = run_dir / "passes"
        final_dir = run_dir / "final"
        logs_dir = run_dir / "logs"

        input_dir.mkdir(parents=True, exist_ok=True)
        passes_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        text = input_path.read_text(encoding=self.config.encoding)
        document = {
            "doc_id": doc_id,
            "title": title,
            "source_path": str(input_path),
            "content_type": "text/plain",
            "encoding": self.config.encoding,
            "text": text,
            "metadata": {
                "author": None,
                "created_at": None,
                "user_goal": user_goal,
            },
        }

        document_schema = load_schema(self.repo_paths.schemas_dir, "document.schema.json")
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
            raise ValueError(f"Chunk validation failed: {chunk_errors}")

        (input_dir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

        normalize_input = {
            "doc_manifest": {"doc_id": doc_id, "title": title},
            "user_goal": user_goal,
            "requested_deliverable": requested_deliverable,
            "user_constraints": [
                "Do not invent facts",
                "Surface uncertainty explicitly",
            ],
        }
        normalize = self.pass_runner.run_model_pass(
            pass_name="00_normalize_request",
            prompt_filename="00_normalize_request.txt",
            schema_filename="00_normalize_request.schema.json",
            input_payload=normalize_input,
            output_path=passes_dir / "00_normalize_request.json",
        )

        extraction_dir = passes_dir / "01_extract_chunk"
        chunk_extractions = []
        for chunk in chunks:
            extraction = self.pass_runner.run_model_pass(
                pass_name="01_extract_chunk",
                prompt_filename="01_extract_chunk.txt",
                schema_filename="01_extract_chunk.schema.json",
                input_payload={"task": normalize, "chunk": chunk},
                output_path=extraction_dir / f"{chunk['chunk_id']}.json",
            )
            chunk_extractions.append(extraction)

        merge = merge_chunk_extractions(doc_id=doc_id, chunk_extractions=chunk_extractions)
        self.pass_runner.write_validated_json(
            schema_filename="02_merge_global.schema.json",
            payload=merge,
            output_path=passes_dir / "02_merge_global.json",
        )

        chunk_summaries = [
            {"chunk_id": item["chunk_id"], "section_role": item["section_role"]}
            for item in chunk_extractions
        ]

        schema_audit = self.pass_runner.run_model_pass(
            pass_name="03_schema_audit",
            prompt_filename="03_schema_audit.txt",
            schema_filename="03_schema_audit.schema.json",
            input_payload={"task": normalize, "merge": merge, "chunk_summaries": chunk_summaries},
            output_path=passes_dir / "03_schema_audit.json",
        )

        dependency_audit = self.pass_runner.run_model_pass(
            pass_name="04_dependency_audit",
            prompt_filename="04_dependency_audit.txt",
            schema_filename="04_dependency_audit.schema.json",
            input_payload={"task": normalize, "merge": merge, "schema_audit": schema_audit},
            output_path=passes_dir / "04_dependency_audit.json",
        )

        assumption_audit = self.pass_runner.run_model_pass(
            pass_name="05_assumption_audit",
            prompt_filename="05_assumption_audit.txt",
            schema_filename="05_assumption_audit.schema.json",
            input_payload={
                "task": normalize,
                "merge": merge,
                "schema_audit": schema_audit,
                "dependency_audit": dependency_audit,
            },
            output_path=passes_dir / "05_assumption_audit.json",
        )

        evidence_audit = self.pass_runner.run_model_pass(
            pass_name="06_evidence_audit",
            prompt_filename="06_evidence_audit.txt",
            schema_filename="06_evidence_audit.schema.json",
            input_payload={
                "merge": merge,
                "schema_audit": schema_audit,
                "dependency_audit": dependency_audit,
                "assumption_audit": assumption_audit,
            },
            output_path=passes_dir / "06_evidence_audit.json",
        )

        synthesis = self.pass_runner.run_model_pass(
            pass_name="07_synthesize",
            prompt_filename="07_synthesize.txt",
            schema_filename="07_synthesize.schema.json",
            input_payload={
                "task": normalize,
                "merge": merge,
                "schema_audit": schema_audit,
                "dependency_audit": dependency_audit,
                "assumption_audit": assumption_audit,
                "evidence_audit": evidence_audit,
            },
            output_path=passes_dir / "07_synthesize.json",
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
        self.pass_runner.write_validated_json(
            schema_filename="08_validate_final.schema.json",
            payload=validation,
            output_path=passes_dir / "08_validate_final.json",
        )

        (final_dir / "final_answer.json").write_text(json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8")
        (final_dir / "final_answer.md").write_text(render_final_answer_markdown(synthesis), encoding="utf-8")

        run_log = [
            f"input={input_path}",
            f"run_id={run_id}",
            f"validation_pass={validation['pass']}",
            f"errors={len(validation['errors'])}",
            f"warnings={len(validation['warnings'])}",
        ]
        (logs_dir / "run.log").write_text("\n".join(run_log) + "\n", encoding="utf-8")

        return run_dir
