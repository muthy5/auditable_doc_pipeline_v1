from __future__ import annotations

import abc
import re
from typing import Any, Dict, List


class LocalLLMBackend(abc.ABC):
    @abc.abstractmethod
    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RuleBasedDemoBackend(LocalLLMBackend):
    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> Dict[str, Any]:
        dispatch = {
            "00_normalize_request": self._normalize_request,
            "01_extract_chunk": self._extract_chunk,
            "classify_document": self._classify_document,
            "03_schema_audit": self._schema_audit,
            "04_dependency_audit": self._dependency_audit,
            "05_assumption_audit": self._assumption_audit,
            "06_evidence_audit": self._evidence_audit,
            "07_synthesize": self._synthesize,
            "09_generate_plan": self._generate_plan,
            "search_queries": self._search_queries,
        }
        if pass_name not in dispatch:
            raise ValueError(f"Unsupported demo pass: {pass_name}")
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


    def _classify_document(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        lower_text = text.lower()

        heuristics = [
            ("legal_contract", ["contract", "agreement", "governing law", "termination", "parties"]),
            ("project_proposal", ["proposal", "problem statement", "deliverables", "success metrics", "scope"]),
            ("medical_protocol", ["protocol", "dosing", "contraindications", "adverse events", "patient"]),
            ("technical_spec", ["technical specification", "architecture", "interfaces", "requirements", "data model"]),
            ("business_plan", ["business plan", "executive summary", "market analysis", "financial projections", "marketing strategy"]),
            ("procedural_plan", ["steps", "ingredients", "materials", "expected output", "procedure"]),
        ]

        best_type = "procedural_plan"
        best_score = 0
        for doc_type, keywords in heuristics:
            score = sum(1 for keyword in keywords if keyword in lower_text)
            if score > best_score:
                best_type = doc_type
                best_score = score

        confidence = "low"
        if best_score >= 3:
            confidence = "high"
        elif best_score >= 1:
            confidence = "medium"

        reason = f"Matched {best_score} keyword signals for {best_type}."
        return {"document_type": best_type, "confidence": confidence, "reason": reason}

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

        template = payload.get("document_type_schema", {})
        document_type = template.get("document_type", payload.get("document_type", "procedural_plan"))
        expected_sections = template.get("expected_sections", [])
        present_sections: list[str] = []
        partial_sections = []
        missing_sections = []
        blocking_gaps = []
        nonblocking_gaps = []

        evidence_text = " ".join(
            materials
            + outputs
            + [step["text"].lower() for step in merge.get("all_steps", [])]
            + [fact.get("text", "").lower() for fact in merge.get("all_explicit_facts", [])]
        )

        section_hints = {
            "executive summary": ["executive summary", "summary"],
            "market analysis": ["market", "competitor", "segment"],
            "financial projections": ["financial", "revenue", "profit", "projection"],
            "operations plan": ["operations", "process", "workflow"],
            "marketing strategy": ["marketing", "go-to-market", "channel"],
            "team": ["team", "roles", "staff"],
            "timeline": ["timeline", "milestone", "schedule"],
            "risks": ["risk", "mitigation"],
            "parties": ["party", "parties"],
            "definitions": ["definition", "defined term"],
            "terms": ["term", "duration"],
            "obligations": ["obligation", "shall"],
            "payment": ["payment", "fee", "invoice"],
            "termination": ["termination", "terminate"],
            "dispute resolution": ["dispute", "arbitration"],
            "governing law": ["governing law", "jurisdiction"],
            "signatures": ["signature", "signed"],
            "problem statement": ["problem", "challenge"],
            "proposed solution": ["solution", "approach"],
            "scope": ["scope", "in scope", "out of scope"],
            "deliverables": ["deliverable", "outcome"],
            "budget": ["budget", "cost"],
            "success metrics": ["kpi", "metric", "success"],
            "indication": ["indication", "condition"],
            "patient selection": ["patient", "eligibility"],
            "procedure steps": ["procedure", "step"],
            "dosing": ["dose", "dosing"],
            "monitoring": ["monitoring", "observe"],
            "adverse events": ["adverse event", "side effect"],
            "contraindications": ["contraindication", "do not use"],
            "follow-up": ["follow-up", "follow up"],
            "overview": ["overview", "introduction"],
            "requirements": ["requirement", "must"],
            "architecture": ["architecture", "component"],
            "interfaces": ["api", "interface"],
            "data model": ["data model", "schema"],
            "security": ["security", "auth", "encryption"],
            "testing": ["test", "validation"],
            "deployment": ["deploy", "release"],
            "objective": ["objective", "goal"],
            "inputs": ["inputs", "input", "materials", "ingredients", "tool", "resource", "prerequisite"],
            "process_steps": ["step", "instructions", "procedure"],
            "outputs": ["output", "result"],
            "constraints": ["constraint", "limit", "requirement", "safety", "risk"],
            "validation_criteria": ["criteria", "quality", "done", "verify", "acceptance"],
        }

        for section in expected_sections:
            hints = section_hints.get(section, [section])
            if any(hint in evidence_text for hint in hints):
                present_sections.append(section)
            else:
                missing_sections.append({"section": section, "reason": f"No evidence found for section '{section}'."})

        if document_type == "procedural_plan" and any("lemon" in item for item in materials) and any("lemonade" in item for item in outputs):
            if not any(keyword in all_steps_text for keyword in ["juice", "squeez", "press", "extract"]):
                blocking_gaps.append(
                    {
                        "gap_id": "gap_schema_001",
                        "section": "ingredient_preparation",
                        "reason": "A transformation step from whole lemons to lemon juice is missing.",
                    }
                )

        if document_type == "procedural_plan" and "validation_criteria" not in present_sections:
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
        schema_audit = payload.get("schema_audit", {})
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
        if not bottom_line_support:
            completeness_claim = next((c for c in evidence.get("claim_registry", []) if "operationally complete" in str(c.get("text", "")).lower()), None)
            if completeness_claim and completeness_claim.get("claim_id"):
                bottom_line_support.append(str(completeness_claim["claim_id"]))

        present_sections = schema_audit.get("present_sections", [])
        missing_sections = schema_audit.get("missing_sections", [])
        organized_structure = [
            {
                "section": "Objective",
                "content": "Assess operational completeness against the selected document schema.",
            },
            {
                "section": "Present content",
                "content": "Detected sections: " + (", ".join(present_sections) if present_sections else "none clearly grounded in extracted evidence."),
            },
            {
                "section": "Missing content",
                "content": "Missing sections: " + (", ".join(str(item.get("section", "unknown")) for item in missing_sections) if missing_sections else "none."),
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


    def _search_queries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task = payload.get("task", {})
        goal = task.get("task", {}).get("primary_goal", "") if isinstance(task, dict) else ""
        document_text = str(payload.get("document_text", ""))
        tokens = [
            token
            for token in re.findall(r"[A-Za-z]{4,}", document_text.lower())
            if token not in {"this", "that", "with", "from", "into", "your", "have", "will", "step", "plan"}
        ]
        unique_tokens: List[str] = []
        for token in tokens:
            if token not in unique_tokens:
                unique_tokens.append(token)
            if len(unique_tokens) >= 3:
                break
        tail = " ".join(unique_tokens)
        base = goal or "document analysis plan"
        queries = [
            f"latest best practices {base}",
            f"checklist to validate {base}",
            f"common risks and safety considerations {base}",
        ]
        if tail:
            queries.append(f"current guidance {tail}")
        return {"queries": queries[:5]}

    def _generate_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        merge = payload["merge"]
        dependency = payload["dependency_audit"]
        materials = merge.get("global_entities", {}).get("materials", []) or merge.get("all_inputs_required", [])
        outputs = merge.get("all_outputs_produced", [])
        input_steps = merge.get("all_steps", [])
        step_text = " ".join(str(step.get("text", "")) for step in input_steps).lower()
        evidence_text = " ".join([*materials, *outputs, step_text]).lower()
        lemonade_context = any(term in evidence_text for term in ["lemon", "lemons", "lemonade"])

        unsupported_demo = not input_steps and not materials and not outputs
        if unsupported_demo:
            return {
                "doc_id": merge["doc_id"],
                "plan": {
                    "objective": {
                        "text": "Summarize available content without fabricating an operational plan.",
                        "support": ["original_document"],
                    },
                    "materials_and_quantities": [],
                    "equipment_required": [],
                    "prerequisites": [],
                    "steps": [],
                    "time_estimates": {
                        "total_estimated": "unknown",
                        "confidence": "unknown",
                    },
                    "warnings_and_safety": [
                        {
                            "text": "Demo backend cannot reliably transform this document type; use Claude or Ollama for full analysis.",
                            "severity": "warning",
                            "support": ["original_document"],
                        }
                    ],
                    "quality_checkpoints": [],
                    "blocking_items": [
                        {
                            "text": "Insufficient procedural structure detected in the uploaded text for reliable demo planning.",
                            "support": ["original_document"],
                        }
                    ],
                    "assumptions_made": [
                        {
                            "text": "Demo mode is limited to simple, explicit procedural inputs.",
                            "support": ["demo_backend"],
                        }
                    ],
                    "cost_indicators": [],
                    "contingencies": [
                        {
                            "if_condition": "Document is non-procedural or ambiguous",
                            "then_action": "Switch to Claude or Ollama backend for full analysis.",
                            "support": ["demo_backend"],
                        }
                    ],
                },
            }

        plan_steps = [
            {
                "step_number": index,
                "text": str(step.get("text", "")).strip() or "Unspecified step",
                "status": "original",
                "support": [str(step.get("step_id", "original_document"))],
            }
            for index, step in enumerate(input_steps, start=1)
        ]

        warnings = [
            {
                "text": "No blocking dependency issues detected.",
                "severity": "info",
                "support": ["dependency_audit"],
            }
        ]

        if lemonade_context and dependency.get("blocking_dependencies"):
            plan_steps.insert(
                1,
                {
                    "step_number": 2,
                    "text": "Cut and juice the lemons into the pitcher.",
                    "status": "added",
                    "support": ["dependency_audit:block_dep_001"],
                    "warning": "This is a required transformation step missing in the original plan.",
                },
            )
            for index, step in enumerate(plan_steps, start=1):
                step["step_number"] = index
            warnings = [
                {
                    "text": "Original plan omitted a juicing step, which blocks successful output.",
                    "severity": "warning",
                    "support": ["block_dep_001"],
                }
            ]

        return {
            "doc_id": merge["doc_id"],
            "plan": {
                "objective": {
                    "text": "Prepare the documented procedure using listed inputs.",
                    "support": ["original_document"],
                },
                "materials_and_quantities": [
                    {"item": str(item), "quantity": "unknown", "source": "stated"}
                    for item in materials
                ],
                "equipment_required": [],
                "prerequisites": [
                    {
                        "text": str(dep.get("before", "Earlier steps")) + " must occur before " + str(dep.get("after", "later steps")) + ".",
                        "support": [str(dep.get("constraint_id", "dependency_audit"))],
                    }
                    for dep in dependency.get("ordering_constraints", [])
                ],
                "steps": plan_steps,
                "time_estimates": {
                    "total_estimated": "unknown",
                    "confidence": "unknown",
                },
                "warnings_and_safety": warnings,
                "quality_checkpoints": [],
                "blocking_items": [
                    {"text": str(dep.get("reason", "Unspecified dependency issue.")), "support": [str(dep.get("dependency_id", "dependency_audit"))]}
                    for dep in dependency.get("blocking_dependencies", [])
                ],
                "assumptions_made": [
                    {
                        "text": "Only explicitly stated steps and materials were used in demo mode.",
                        "support": ["demo_backend"],
                    }
                ],
                "cost_indicators": [
                    {"item": str(item), "cost": "unknown", "source": "unknown"}
                    for item in materials
                ],
                "contingencies": [],
            },
        }
