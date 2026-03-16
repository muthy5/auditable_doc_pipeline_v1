from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

from app_utils import (
    PASS_SEQUENCE,
    collect_pass_outputs,
    collect_run_report,
    format_item,
    parse_final_sections,
    parse_plan_output,
    read_final_markdown,
)
from src.config import PipelineConfig
from src.pipeline import AuditablePipeline

st.set_page_config(page_title="Auditable Document Pipeline", page_icon="📄", layout="wide")
st.title("Auditable Document Pipeline")

with st.sidebar:
    st.header("Run Settings")
    backend = st.selectbox("Backend", ["demo", "ollama", "claude"], index=0)

    claude_api_key = ""
    claude_model = "claude-sonnet-4-20250514"
    if backend == "claude":
        claude_api_key = st.text_input("Claude API Key", type="password")
        claude_model = st.text_input("Claude model", value="claude-sonnet-4-20250514")

    ollama_base_url = "http://127.0.0.1:11434"
    ollama_model = ""
    if backend == "ollama":
        ollama_base_url = st.text_input("Ollama base URL", value="http://127.0.0.1:11434")
        ollama_model = st.text_input("Ollama model name", value="")

    strict_mode = st.toggle("Strict mode", value=False)
    user_goal = st.text_area(
        "User goal",
        value="Identify missing information and organize the document into an actionable structure.",
        height=140,
    )

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
run_clicked = st.button("Run Pipeline", type="primary", disabled=uploaded_file is None)

if run_clicked:
    if uploaded_file is None:
        st.error("Please upload a .txt file before running the pipeline.")
    elif backend == "claude" and not claude_api_key:
        st.error("Please provide a Claude API key.")
    elif backend == "ollama" and not ollama_model:
        st.error("Please provide an Ollama model name.")
    else:
        try:
            with tempfile.TemporaryDirectory(prefix="auditable_pipeline_") as temp_dir:
                temp_root = Path(temp_dir)
                input_path = temp_root / uploaded_file.name
                input_path.write_bytes(uploaded_file.getvalue())

                config = PipelineConfig(
                    claude_api_key=claude_api_key,
                    claude_model=claude_model,
                    ollama_base_url=ollama_base_url,
                    ollama_model=ollama_model,
                )

                pipeline = AuditablePipeline(repo_root=Path(__file__).parent, backend_name=backend, config=config)

                result: dict[str, Path | Exception | None] = {"run_dir": None, "error": None}

                def _execute_pipeline() -> None:
                    try:
                        run_dir = pipeline.run(
                            input_path=input_path,
                            runs_dir=temp_root / "runs",
                            user_goal=user_goal,
                            strict=strict_mode,
                        )
                        result["run_dir"] = run_dir
                    except Exception as exc:  # noqa: BLE001 - surfaced to Streamlit users
                        result["error"] = exc

                thread = threading.Thread(target=_execute_pipeline, daemon=True)
                thread.start()

                progress = st.progress(0, text="Starting pipeline...")
                status_placeholder = st.empty()
                pass_completion = {name: False for name in PASS_SEQUENCE}
                passes_dir = temp_root / "runs"

                while thread.is_alive():
                    run_dirs = sorted(passes_dir.glob("*/passes")) if passes_dir.exists() else []
                    if run_dirs:
                        active_passes_dir = run_dirs[-1]
                        for pass_name in PASS_SEQUENCE:
                            if pass_name == "01_extract_chunk":
                                pass_completion[pass_name] = (active_passes_dir / "01_extract_chunk").exists() and any(
                                    (active_passes_dir / "01_extract_chunk").glob("*.json")
                                )
                            else:
                                pass_completion[pass_name] = (active_passes_dir / f"{pass_name}.json").exists()

                    completed = sum(1 for done in pass_completion.values() if done)
                    current = PASS_SEQUENCE[min(completed, len(PASS_SEQUENCE) - 1)]
                    progress.progress(completed / len(PASS_SEQUENCE), text=f"Running {current}...")
                    status_placeholder.info(f"Completed {completed}/{len(PASS_SEQUENCE)} passes")
                    time.sleep(0.25)

                thread.join()

                error = result["error"]
                if error:
                    st.error(f"Pipeline failed: {error}")
                else:
                    run_dir = result["run_dir"]
                    if run_dir is None:
                        st.error("Pipeline did not produce a run directory.")
                    else:
                        progress.progress(1.0, text="Pipeline complete")
                        status_placeholder.success("Pipeline completed successfully")

                        sections = parse_final_sections(run_dir)
                        plan = parse_plan_output(run_dir)
                        pass_outputs = collect_pass_outputs(run_dir)
                        run_report = collect_run_report(run_dir)
                        final_markdown = read_final_markdown(run_dir)

                        st.subheader("Results")
                        summary_tab, plan_tab = st.tabs(["Final Answer", "Generated Plan"])

                        with summary_tab:
                            st.markdown("### 🟩 Verified Content")
                            st.success("\n".join(format_item(item) for item in sections["verified_content"]) or "- None")

                            st.markdown("### 🟥 Missing Information")
                            st.error("\n".join(format_item(item) for item in sections["missing_information"]) or "- None")

                            st.markdown("### 🟧 Blocking Dependencies")
                            st.warning("\n".join(format_item(item) for item in sections["dependencies"]) or "- None")

                            st.markdown("### 🟨 Assumptions")
                            st.warning("\n".join(format_item(item) for item in sections["assumptions"]) or "- None")

                            st.markdown("### 🟦 Uncertainties")
                            st.info("\n".join(format_item(item) for item in sections["uncertainties"]) or "- None")

                            bottom_line = sections["bottom_line"] or "No bottom line generated."
                            st.markdown("### Bottom Line")
                            st.markdown(
                                f"<div style='font-size:1.2rem;padding:1rem;border-left:6px solid #3b82f6;background:#eef6ff;'>"
                                f"<strong>{bottom_line}</strong></div>",
                                unsafe_allow_html=True,
                            )

                        with plan_tab:
                            if not plan:
                                st.info("No generated plan available.")
                            else:
                                st.markdown("### Objective")
                                st.write(plan.get("objective", {}).get("text", ""))

                                st.markdown("### Materials")
                                st.table(plan.get("materials_and_quantities", []))

                                st.markdown("### Steps")
                                color_map = {"original": "#16a34a", "added": "#2563eb", "reordered": "#f97316"}
                                for step in plan.get("steps", []):
                                    color = color_map.get(step.get("status"), "#64748b")
                                    st.markdown(
                                        f"<div style='padding:0.5rem;margin-bottom:0.25rem;border-left:6px solid {color};'>"
                                        f"<strong>{step.get('step_number', '?')}.</strong> {step.get('text', '')} "
                                        f"<em>({step.get('status', 'unknown')})</em></div>",
                                        unsafe_allow_html=True,
                                    )

                                st.markdown("### Warnings & Safety")
                                severity_colors = {"critical": "#dc2626", "warning": "#f97316", "info": "#0ea5e9"}
                                for warning in plan.get("warnings_and_safety", []):
                                    color = severity_colors.get(warning.get("severity"), "#64748b")
                                    st.markdown(
                                        f"<div style='padding:0.5rem;margin-bottom:0.25rem;border-left:6px solid {color};'>"
                                        f"<strong>{warning.get('severity', 'info').upper()}:</strong> {warning.get('text', '')}</div>",
                                        unsafe_allow_html=True,
                                    )

                        with st.expander("Raw Pass Outputs"):
                            st.json(pass_outputs)

                        with st.expander("Run Report"):
                            st.json(run_report)

                        st.download_button(
                            "Download final_answer.md",
                            data=final_markdown,
                            file_name="final_answer.md",
                            mime="text/markdown",
                        )
        except Exception as exc:  # noqa: BLE001 - catch top-level app exceptions
            st.error(f"Unexpected error: {exc}")
