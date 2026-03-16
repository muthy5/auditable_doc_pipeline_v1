from __future__ import annotations

import abc
import re
from typing import Any, Dict, List

from .exceptions import BackendError


class LocalLLMBackend(abc.ABC):
    """Abstract interface for local JSON-producing backends."""
    @abc.abstractmethod
    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RuleBasedDemoBackend(LocalLLMBackend):
    """Rule-based backend used for deterministic local demos."""
    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        dispatch = {
            "00_normalize_request": self._normalize_request,
            "01_extract_chunk": self._extract_chunk,
            "03_schema_audit": self._schema_audit,
            "04_dependency_audit": self._dependency_audit,
            "05_assumption_audit": self._assumption_audit,
            "06_evidence_audit": self._evidence_audit,
            "07_synthesize": self._synthesize,
        }
        if pass_name not in dispatch:
            raise BackendError(f"Unsupported demo pass: {pass_name}")
        return dispatch[pass_name](payload)

    def _normalize_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        goal = payload["user_goal"]
        return {
            "doc_id": payload["doc_manifest"]["doc_id"],
            "task": {
                "primary_goal": goal,
                "deliverable_type": payload["requested_deliverable"],
                "domain": "procedural_plan" if "plan" in goal.lower() else "unspecified",
                "audience": "analyst",
                "jurisdiction": "unspecified",
                "timeframe": "unspecified",
            },
            "questions_to_answer": [
                "What information is present?",
                "What information is missing?",
                "What dependencies exist?",
                "What assumptions and uncertainties exist?",
                "Is the document operationally complete?",
            ],
            "explicit_constraints": payload.get("user_constraints", []),
            "missing_critical_inputs": [],
            "blocked": False,
            "blocking_reason": None,
        }

    def _extract_chunk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        chunk = payload["chunk"]
        text = chunk["text"]
        heading = " > ".join(chunk.get("heading_path", []))
        lower_text = text.lower()

        section_role = "general_section"
        if "ingredient" in lower_text or "materials" in lower_text:
            section_role = "materials_specification"
        elif "plan:" in lower_text or "steps" in lower_text:
            section_role = "process_steps"
        elif "goal:" in lower_text or "objective" in lower_text:
            section_role = "objective"
        elif "output" in lower_text:
            section_role = "outputs"

        explicit_facts: List[Dict[str, Any]] = []
        for label in ["title", "goal", "expected output"]:
            match = re.search(rf"{label}:\s*(.+)", text, flags=re.IGNORECASE)
            if match:
                quote = match.group(0).strip()
                explicit_facts.append(
                    {
                        "fact_id": f"f_{chunk['chunk_id']}_{len(explicit_facts)+1:03d}",
                        "text": f"{label.title()} is {match.group(1).strip()}",
                        "quote": quote,
                        "source_span": {"start_char": match.start(), "end_char": match.end()},
                    }
                )

        material_lines = re.findall(r"^-\s+(.+)$", text, flags=re.MULTILINE)
        materials: List[str] = []
        steps: List[Dict[str, Any]] = []
        if "ingredients:" in lower_text or "materials:" in lower_text:
            materials.extend(material_lines)

        step_matches = re.finditer(r"^(\d+)\.\s+(.+)$", text, flags=re.MULTILINE)
        for match in step_matches:
            steps.append(
                {
                    "step_id": f"s_{chunk['chunk_id']}_{int(match.group(1)):03d}",
                    "ordinal": int(match.group(1)),
                    "text": match.group(2).strip(),
                }
            )

        outputs = []
        output_match = re.search(r"expected output:\s*(.+)", text, flags=re.IGNORECASE)
        if output_match:
            outputs.append(output_match.group(1).strip().rstrip("."))

        missing_information_signals: List[str] = []
        if any("lemon" in item.lower() for item in materials) and outputs and any("lemonade" in o.lower() for o in outputs):
            step_text = " ".join(step["text"].lower() for step in steps)
            if not any(keyword in step_text for keyword in ["juice", "squeez", "press", "extract"]):
                missing_information_signals.append("No step extracts juice from the lemons.")

        claims = [
            {
                "claim_id": f"c_{chunk['chunk_id']}_001",
                "text": f"This chunk functions as {section_role}.",
            }
        ]

        entities = {
            "materials": materials,
            "equipment": [],
            "people_roles": [],
            "documents": [],
            "locations": [],
            "constraints": [],
        }

        cross_refs: List[Dict[str, Any]] = []
        if "previous module" in lower_text or "above" in lower_text or "below" in lower_text:
            cross_refs.append({"ref_text": "cross-reference", "target_known": False, "target_label": None})

        return {
            "chunk_id": chunk["chunk_id"],
            "section_role": section_role,
            "summary": heading if heading else "General chunk content",
            "explicit_facts": explicit_facts,
            "claims": claims,
            "entities": entities,
            "steps": steps,
            "dependencies_mentioned": [],
            "inputs_required": materials,
            "outputs_produced": outputs,
            "assumptions_found": [],
            "uncertainties_found": [],
            "missing_information_signals": missing_information_signals,
            "cross_references": cross_refs,
            "defined_terms": [],
            "undefined_terms": [],
            "risks_mentioned": [],
            "contradictions_local": [],
            "needs_followup": bool(missing_information_signals),
        }

    def _schema_audit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        merge = payload["merge"]
        materials = [m.lower() for m in merge.get("global_entities", {}).get("materials", [])]
        outputs = [o.lower() for o in merge.get("all_outputs_produced", [])]
        all_steps_text = " ".join(step["text"].lower() for step in merge.get("all_steps", []))

        document_type = "procedural_plan"
        expected_sections = [
            "objective",
            "materials",
            "ingredient_preparation",
            "process_steps",
            "outputs",
            "validation_criteria",
        ]
        present_sections = []
        partial_sections = []
        missing_sections = []
        blocking_gaps = []
        nonblocking_gaps = []

        chunk_roles = {entry["section_role"] for entry in payload.get("chunk_summaries", [])}
        if "objective" in chunk_roles:
            present_sections.append("objective")
        if any(materials):
            present_sections.append("materials")
        if merge.get("all_steps"):
            present_sections.append("process_steps")
        if outputs:
            present_sections.append("outputs")

        if any("lemon" in item for item in materials) and any("lemonade" in item for item in outputs):
            if any(keyword in all_steps_text for keyword in ["juice", "squeez", "press", "extract"]):
                present_sections.append("ingredient_preparation")
            else:
                missing_sections.append(
                    {
                        "section": "ingredient_preparation",
                        "reason": "Whole lemons are present, but no extraction step appears.",
                    }
                )
                blocking_gaps.append(
                    {
                        "gap_id": "gap_schema_001",
                        "section": "ingredient_preparation",
                        "reason": "A transformation step from whole lemons to lemon juice is missing.",
                    }
                )

        if "validation_criteria" not in present_sections:
            nonblocking_gaps.append(
                {
                    "gap_id": "gap_schema_002",
                    "section": "validation_criteria",
                    "reason": "No explicit quality or completion criteria were found.",
                }
            )

        return {
            "doc_id": merge["doc_id"],
            "document_type": document_type,
            "expected_sections": expected_sections,
            "present_sections": sorted(set(present_sections)),
            "partial_sections": partial_sections,
            "missing_sections": missing_sections,
            "extraneous_sections": [],
            "blocking_gaps": blocking_gaps,
            "nonblocking_gaps": nonblocking_gaps,
        }

    def _dependency_audit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        merge = payload["merge"]
        materials = [m.lower() for m in merge.get("global_entities", {}).get("materials", [])]
        outputs = [o.lower() for o in merge.get("all_outputs_produced", [])]
        step_text = " ".join(step["text"].lower() for step in merge.get("all_steps", []))
        missing_prerequisites = []
        ordering_constraints = []
        blocking_dependencies = []
        required_prerequisites = []
        dependency_chains = []
        dangling_refs = []
        internal_conflicts = []

        if any("lemon" in item for item in materials):
            required_prerequisites.append("ingredient preparation must occur before mixing")
            ordering_constraints.append({"constraint_id": "ord_001", "before": "ingredient preparation", "after": "mixing"})
        if merge.get("all_steps"):
            dependency_chains.append(
                {
                    "chain_id": "dep_001",
                    "steps": ["ingredient preparation", "mixing", "chilling", "serving"],
                }
            )

        if any("lemon" in item for item in materials) and any("lemonade" in item for item in outputs):
            if not any(keyword in step_text for keyword in ["juice", "squeez", "press", "extract"]):
                missing_prerequisites.append(
                    {
                        "item": "lemon juice extraction step",
                        "why_required": "Whole lemons do not become lemonade without a transformation step.",
                    }
                )
                blocking_dependencies.append(
                    {
                        "dependency_id": "block_dep_001",
                        "reason": "The plan never turns whole lemons into usable lemon juice.",
                    }
                )

        for ref in merge.get("cross_reference_graph", []):
            dangling_refs.append(
                {
                    "ref_text": ref["ref_text"],
                    "source_chunk_id": ref["source_chunk_id"],
                }
            )

        return {
            "doc_id": merge["doc_id"],
            "required_prerequisites": required_prerequisites,
            "missing_prerequisites": missing_prerequisites,
            "ordering_constraints": ordering_constraints,
            "dangling_references": dangling_refs,
            "internal_conflicts": internal_conflicts,
            "dependency_chains": dependency_chains,
            "blocking_dependencies": blocking_dependencies,
        }

    def _assumption_audit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        merge = payload["merge"]
        dependency = payload["dependency_audit"]
        implicit_assumptions_found = []
        uncertainty_points = []
        overconfidence_flags = []
        context_dependencies = []
        blocking_assumptions = []

        if dependency.get("blocking_dependencies"):
            implicit_assumptions_found.append(
                {
                    "assumption_id": "assume_001",
                    "text": "The reader already knows to juice or squeeze the lemons.",
                }
            )
            blocking_assumptions.append(
                {
                    "assumption_id": "block_assume_001",
                    "reason": "The plan depends on external knowledge for a required transformation step.",
                }
            )
            overconfidence_flags.append(
                {
                    "flag_id": "over_001",
                    "text": "The plan presents itself as complete even though a required preparation step is absent.",
                }
            )

        if merge.get("cross_reference_graph"):
            uncertainty_points.append(
                {
                    "uncertainty_id": "unc_001",
                    "text": "At least one cross-reference remains unresolved.",
                }
            )

        return {
            "doc_id": merge["doc_id"],
            "explicit_assumptions": [],
            "implicit_assumptions_found": implicit_assumptions_found,
            "scope_errors": [],
            "context_dependencies": context_dependencies,
            "uncertainty_points": uncertainty_points,
            "overconfidence_flags": overconfidence_flags,
            "blocking_assumptions": blocking_assumptions,
        }

    def _evidence_audit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        merge = payload["merge"]
        schema_audit = payload["schema_audit"]
        claim_registry = []
        counts = {"supported": 0, "inferred": 0, "speculative": 0, "unknown": 0}

        for fact in merge.get("all_explicit_facts", []):
            claim_registry.append(
                {
                    "claim_id": f"claim_{len(claim_registry)+1:03d}",
                    "text": fact["text"],
                    "status": "supported",
                    "basis": [
                        {
                            "type": "explicit_fact",
                            "source_chunk_ids": fact["source_chunk_ids"],
                            "source_refs": fact["source_refs"],
                        }
                    ],
                    "reason": "Directly grounded in document text.",
                }
            )
            counts["supported"] += 1

        completeness_status = "unknown" if schema_audit.get("blocking_gaps") else "inferred"
        claim_registry.append(
            {
                "claim_id": f"claim_{len(claim_registry)+1:03d}",
                "text": "The document is operationally complete.",
                "status": completeness_status,
                "basis": [],
                "reason": "Completeness cannot be established while blocking gaps remain." if completeness_status == "unknown" else "No blocking gaps were identified.",
            }
        )
        counts[completeness_status] += 1

        return {
            "doc_id": merge["doc_id"],
            "claim_registry": claim_registry,
            "counts": counts,
        }

    def _synthesize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task = payload["task"]
        schema_audit = payload["schema_audit"]
        dependency_audit = payload["dependency_audit"]
        assumption_audit = payload["assumption_audit"]
        evidence = payload["evidence_audit"]

        verified_content = []
        for claim in evidence.get("claim_registry", []):
            if claim["status"] == "supported":
                verified_content.append({"text": claim["text"], "support": [claim["claim_id"]]})

        missing_information = []
        for gap in schema_audit.get("blocking_gaps", []):
            missing_information.append({"text": gap["reason"], "support": [gap["gap_id"]]})
        for gap in schema_audit.get("nonblocking_gaps", []):
            missing_information.append({"text": gap["reason"], "support": [gap["gap_id"]]})

        dependencies = []
        for dep in dependency_audit.get("blocking_dependencies", []):
            dependencies.append({"text": dep["reason"], "support": [dep["dependency_id"]]})
        for dep in dependency_audit.get("ordering_constraints", []):
            dependencies.append(
                {
                    "text": f"{dep['before']} must occur before {dep['after']}.",
                    "support": [dep.get("constraint_id")] if dep.get("constraint_id") else [],
                }
            )

        assumptions = []
        for item in assumption_audit.get("implicit_assumptions_found", []):
            assumptions.append({"text": item["text"], "support": [item["assumption_id"]]})

        uncertainties = []
        for item in assumption_audit.get("uncertainty_points", []):
            uncertainties.append({"text": item["text"], "support": [item["uncertainty_id"]]})

        incomplete = bool(
            schema_audit.get("blocking_gaps")
            or dependency_audit.get("blocking_dependencies")
            or assumption_audit.get("blocking_assumptions")
        )
        bottom_line_support: List[str] = []
        bottom_line_support.extend(g["gap_id"] for g in schema_audit.get("blocking_gaps", []))
        bottom_line_support.extend(d["dependency_id"] for d in dependency_audit.get("blocking_dependencies", []))
        bottom_line_support.extend(a["assumption_id"] for a in assumption_audit.get("blocking_assumptions", []))

        organized_structure = [
            {
                "section": "Objective",
                "content": "The document aims to describe a procedural plan and expected output.",
            },
            {
                "section": "Present content",
                "content": "The document includes ingredients, a simple step list, and an expected output.",
            },
            {
                "section": "Missing content",
                "content": "The document omits at least one required transformation or quality-control element.",
            },
        ]

        return {
            "doc_id": payload["merge"]["doc_id"],
            "final_answer": {
                "goal": task["task"]["primary_goal"],
                "verified_content": verified_content,
                "missing_information": missing_information,
                "dependencies": dependencies,
                "assumptions": assumptions,
                "uncertainties": uncertainties,
                "organized_structure": organized_structure,
                "bottom_line": {
                    "text": "The document is not operationally complete on the available evidence."
                    if incomplete
                    else "The document appears operationally complete on the available evidence.",
                    "support": bottom_line_support,
                },
            },
        }
