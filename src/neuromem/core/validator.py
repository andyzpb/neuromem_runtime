from __future__ import annotations

from dataclasses import dataclass

from neuromem.core.policy import MemoryPolicy, ValidatedPolicy


@dataclass(slots=True)
class PolicyValidator:
    min_write_confidence: float = 0.55

    def validate(self, policy: MemoryPolicy, context: dict[str, object] | None = None) -> ValidatedPolicy:
        context = context or {}
        rejected: list[str] = []
        approved_actions: list[str] = []

        if policy.retrieval.enabled:
            if not policy.retrieval.query.strip():
                rejected.append("retrieval query is required when retrieval is enabled")
            else:
                approved_actions.append("RETRIEVE")

        write = policy.write
        if write.operation in {"ADD", "UPDATE"}:
            if write.confidence < self.min_write_confidence:
                rejected.append("write confidence below threshold")
            if not write.evidence_ids:
                rejected.append("write requires evidence ids")
            if not write.content and write.operation == "ADD":
                rejected.append("ADD requires content")
            if write.operation == "UPDATE" and not write.target_memory_id:
                rejected.append("UPDATE requires target memory id")
            if not rejected:
                approved_actions.append(write.operation)
        elif write.operation == "LINK":
            if not write.target_memory_id or not write.evidence_ids:
                rejected.append("LINK requires target memory id and evidence ids")
            else:
                approved_actions.append("LINK")
        elif write.operation != "NOOP":
            rejected.append(f"unsupported write operation: {write.operation}")

        forget = policy.forget
        if forget.operation == "DELETE_REQUEST":
            if context.get("user_explicit_delete_request") is not True:
                rejected.append("DELETE_REQUEST requires explicit user authorization")
            elif not forget.target_memory_id:
                rejected.append("DELETE_REQUEST requires target memory id")
            else:
                approved_actions.append("DELETE_REQUEST")
        elif forget.operation in {"INVALIDATE", "INHIBIT", "ARCHIVE", "DECAY"}:
            if not forget.target_memory_id:
                rejected.append(f"{forget.operation} requires target memory id")
            if not forget.reason:
                rejected.append(f"{forget.operation} requires a reason")
            if forget.target_memory_id and forget.reason:
                approved_actions.append(forget.operation)
        elif forget.operation != "NOOP":
            rejected.append(f"unsupported forget operation: {forget.operation}")

        consolidation = policy.consolidation
        if consolidation.enabled:
            if not consolidation.cluster_ids:
                rejected.append("consolidation requires cluster ids")
            if consolidation.target_type not in {"semantic", "procedural", "schema"}:
                rejected.append("consolidation target type must be semantic, procedural, or schema")
            if consolidation.objective is None:
                rejected.append("consolidation requires objective")
            if not rejected:
                approved_actions.append("CONSOLIDATE")

        return ValidatedPolicy(policy=policy, approved=not rejected, approved_actions=approved_actions, rejected_reasons=rejected)
