from __future__ import annotations

from dataclasses import dataclass

from neuromem.core.policy import MemoryPolicy
from neuromem.stores.base import MemoryStore
from neuromem_runtime.lifecycle import LifecycleStateMachine
from neuromem_runtime.policy_v2 import ProposedDelta, ValidatedMutation, ValidationStep


@dataclass(slots=True)
class ValidationContext:
    store: MemoryStore | None = None
    ledger: object | None = None
    phase: str = "mutation"
    authorize_delete: bool = False
    user_id: str | None = None
    namespace: str = "default"
    historical: bool = False
    post_commit: bool = False
    affected_memory_ids: list[str] | None = None
    allow_cross_namespace: bool = False


class MutationValidator:
    name = "MutationValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        return ValidationStep(name=self.name, passed=True)


class SchemaValidator(MutationValidator):
    name = "SchemaValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        valid = policy.write.operation in {"ADD", "LINK", "NOOP"} and policy.forget.operation in {"NOOP", "DECAY", "INHIBIT", "INVALIDATE", "ARCHIVE"}
        return ValidationStep(name=self.name, passed=valid, reason="" if valid else "unsupported operation")


class EvidenceValidator(MutationValidator):
    name = "EvidenceValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if policy.write.operation in {"ADD", "UPDATE", "LINK"} and not policy.write.evidence_ids:
            return ValidationStep(name=self.name, passed=False, reason="write mutations require evidence ids")
        return ValidationStep(name=self.name, passed=True)


class ProvenanceValidator(MutationValidator):
    name = "ProvenanceValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if policy.write.operation not in {"ADD", "UPDATE", "LINK"}:
            return ValidationStep(name=self.name, passed=True)
        ledger = context.ledger
        if ledger is None:
            return ValidationStep(name=self.name, passed=True)
        missing: list[str] = []
        for evidence_id in policy.write.evidence_ids:
            exists = False
            if hasattr(ledger, "get_experience"):
                exists = getattr(ledger, "get_experience")(evidence_id, namespace=context.namespace) is not None
            if not exists and context.store is not None:
                exists = context.store.get_memory(evidence_id) is not None
            if not exists:
                missing.append(evidence_id)
        if missing:
            return ValidationStep(name=self.name, passed=False, reason=f"unknown evidence ids: {', '.join(missing)}")
        return ValidationStep(name=self.name, passed=True)


class TemporalValidator(MutationValidator):
    name = "TemporalValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if policy.write.operation != "UPDATE" or not policy.write.target_memory_id or context.store is None:
            return ValidationStep(name=self.name, passed=True)
        item = context.store.get_memory(policy.write.target_memory_id)
        if item is None:
            return ValidationStep(name=self.name, passed=False, reason="target memory not found")
        if item.maturity in {"obsolete", "archived", "deleted", "inhibited"} and not context.historical:
            return ValidationStep(name=self.name, passed=False, reason=f"cannot update {item.maturity} memory without historical intent")
        return ValidationStep(name=self.name, passed=True)


class ConflictValidator(MutationValidator):
    name = "ConflictValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        content = (policy.write.content or "").lower()
        if policy.write.operation in {"ADD", "UPDATE"} and "contradicts" in content and "supersede" not in policy.reason.lower():
            return ValidationStep(name=self.name, passed=False, reason="contradictory write requires supersede/invalidate rationale")
        return ValidationStep(name=self.name, passed=True)


class PrivacyAclValidator(MutationValidator):
    name = "PrivacyAclValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        target_id = policy.write.target_memory_id or policy.forget.target_memory_id
        if not target_id or context.store is None:
            return ValidationStep(name=self.name, passed=True)
        item = context.store.get_memory(target_id)
        if item is None:
            return ValidationStep(name=self.name, passed=False, reason="target memory not found")
        if item.privacy_level in {"user", "sensitive"} and item.acl:
            if context.user_id is None or context.user_id not in item.acl:
                return ValidationStep(name=self.name, passed=False, reason="user is not authorized for private memory")
        return ValidationStep(name=self.name, passed=True)


class NamespaceScopeValidator(MutationValidator):
    name = "NamespaceScopeValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if context.allow_cross_namespace or context.store is None:
            return ValidationStep(name=self.name, passed=True)
        ids = [value for value in [policy.write.target_memory_id, policy.forget.target_memory_id] if value]
        for memory_id in ids:
            item = context.store.get_memory(memory_id)
            if item is None:
                return ValidationStep(name=self.name, passed=False, reason="target memory not found")
            if item.namespace != context.namespace:
                return ValidationStep(name=self.name, passed=False, reason=f"target memory outside namespace: {memory_id}")
        for evidence_id in policy.write.evidence_ids:
            item = context.store.get_memory(evidence_id)
            if item is not None and item.namespace != context.namespace:
                return ValidationStep(name=self.name, passed=False, reason=f"evidence memory outside namespace: {evidence_id}")
        return ValidationStep(name=self.name, passed=True)


class DeletionGuardValidator(MutationValidator):
    name = "DeletionGuardValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if policy.forget.operation == "DELETE_REQUEST":
            return ValidationStep(name=self.name, passed=False, reason="DELETE_REQUEST is not supported by append-only runtime; use suppression or redaction evidence")
        return ValidationStep(name=self.name, passed=True)


class PoisoningRiskValidator(MutationValidator):
    name = "PoisoningRiskValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        content = (policy.write.content or "").lower()
        suspicious = ["ignore previous", "override memory", "always trust this unverified", "delete audit"]
        if any(term in content for term in suspicious):
            return ValidationStep(name=self.name, passed=False, reason="possible memory poisoning instruction")
        return ValidationStep(name=self.name, passed=True)


class WriteGateConsistencyValidator(MutationValidator):
    name = "WriteGateConsistencyValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        gate = _write_gate(policy)
        if not gate:
            if context.phase == "after_step" and policy.source == "small_llm" and policy.write.operation == "NOOP" and policy.forget.operation == "NOOP" and not policy.consolidation.enabled:
                return ValidationStep(name=self.name, passed=False, reason="after_step small_llm NOOP requires write_gate")
            return ValidationStep(name=self.name, passed=True)
        decision = str(gate.get("decision") or "").lower()
        rationale = str(gate.get("rationale") or policy.reason or "").lower()
        writes = policy.write.operation in {"ADD", "UPDATE", "LINK"}
        has_content = bool((policy.write.content or "").strip() or policy.write.target_memory_id)
        has_evidence = bool(policy.write.evidence_ids)
        if decision in {"noop", "defer"} and not rationale.strip():
            return ValidationStep(name=self.name, passed=False, reason="write_gate noop/defer requires rationale")
        if str(gate.get("policy_schema_failure") or "").strip():
            return ValidationStep(name=self.name, passed=False, reason=str(gate["policy_schema_failure"]))
        if decision == "commit":
            if not writes:
                return ValidationStep(name=self.name, passed=False, reason="write_gate commit requires ADD, UPDATE, or LINK")
            if not has_content:
                return ValidationStep(name=self.name, passed=False, reason="write_gate commit requires canonical content or target memory")
            if not has_evidence:
                return ValidationStep(name=self.name, passed=False, reason="write_gate commit requires evidence ids")
        if decision in {"noop", "defer"} and writes:
            return ValidationStep(name=self.name, passed=False, reason="write_gate noop/defer conflicts with durable write mutation")
        if not writes and _claims_durable_value(rationale):
            return ValidationStep(name=self.name, passed=False, reason="write rationale claims durable value but policy is NOOP")
        return ValidationStep(name=self.name, passed=True)


class LifecycleTransitionValidator(MutationValidator):
    name = "LifecycleTransitionValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if policy.forget.operation == "NOOP" or not policy.forget.target_memory_id or context.store is None:
            return ValidationStep(name=self.name, passed=True)
        item = context.store.get_memory(policy.forget.target_memory_id)
        if item is None:
            return ValidationStep(name=self.name, passed=False, reason="target memory not found")
        target_state = {
            "DECAY": item.maturity,
            "INHIBIT": "inhibited",
            "INVALIDATE": "obsolete",
            "ARCHIVE": "archived",
            "DELETE_REQUEST": "deleted",
        }.get(policy.forget.operation, item.maturity)
        machine = LifecycleStateMachine()
        valid = machine.validate_transition(item.maturity, target_state)
        return ValidationStep(name=self.name, passed=valid, reason="" if valid else f"invalid lifecycle transition: {item.maturity} -> {target_state}")


class IndexConsistencyValidator(MutationValidator):
    name = "IndexConsistencyValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if not context.post_commit or context.store is None:
            return ValidationStep(name=self.name, passed=True)
        search = getattr(context.store, "search_memory_cards", None)
        if search is None:
            return ValidationStep(name=self.name, passed=True)
        missing: list[str] = []
        for memory_id in context.affected_memory_ids or []:
            item = context.store.get_memory(memory_id)
            if item is None or item.maturity == "deleted":
                continue
            results = search(item.content[:80], namespace=item.namespace, limit=20)
            if memory_id not in {result_id for result_id, _score in results}:
                missing.append(memory_id)
        if missing:
            return ValidationStep(name=self.name, passed=False, reason=f"missing memory card index rows: {', '.join(missing)}")
        return ValidationStep(name=self.name, passed=True)


class ValidatorStack:
    def __init__(self, validators: list[MutationValidator] | None = None) -> None:
        self.validators = validators or [
            SchemaValidator(),
            EvidenceValidator(),
            ProvenanceValidator(),
            TemporalValidator(),
            ConflictValidator(),
            NamespaceScopeValidator(),
            PrivacyAclValidator(),
            DeletionGuardValidator(),
            PoisoningRiskValidator(),
            WriteGateConsistencyValidator(),
            LifecycleTransitionValidator(),
            IndexConsistencyValidator(),
        ]

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidatedMutation:
        trace = [validator.validate(policy, context) for validator in self.validators]
        approved = all(step.passed for step in trace)
        risk_score = min(1.0, 0.1 + 0.15 * sum(1 for step in trace if not step.passed))
        delta = ProposedDelta(
            operation=policy.write.operation if policy.write.operation != "NOOP" else policy.forget.operation,
            target_memory_id=policy.write.target_memory_id or policy.forget.target_memory_id,
            value=policy.write.content,
            reason=policy.reason,
        )
        return ValidatedMutation(
            approved=approved,
            approved_deltas=[delta] if approved else [],
            rejected_deltas=[] if approved else [delta],
            required_human_review=any(step.name in {"PrivacyAclValidator", "DeletionGuardValidator"} and not step.passed for step in trace),
            risk_score=risk_score if approved else max(0.75, risk_score),
            validator_trace=trace,
        )


__all__ = [
    "ConflictValidator",
    "DeletionGuardValidator",
    "EvidenceValidator",
    "IndexConsistencyValidator",
    "LifecycleTransitionValidator",
    "MutationValidator",
    "NamespaceScopeValidator",
    "PoisoningRiskValidator",
    "PrivacyAclValidator",
    "ProvenanceValidator",
    "SchemaValidator",
    "TemporalValidator",
    "ValidationContext",
    "ValidatorStack",
    "WriteGateConsistencyValidator",
]


def _write_gate(policy: MemoryPolicy) -> dict[str, object]:
    value = policy.write_gate
    return dict(value) if isinstance(value, dict) else {}


def _claims_durable_value(rationale: str) -> bool:
    lowered = rationale.lower()
    negated = [
        "no durable",
        "not durable",
        "not a durable",
        "not new durable",
        "no long-term",
        "not long-term",
        "no future",
        "low future",
        "not useful",
        "no reusable",
        "not reusable",
        "没有长期",
        "不需要长期",
        "不值得记住",
        "不用记住",
    ]
    if any(phrase in lowered for phrase in negated):
        return False
    durable_words = ["durable", "useful", "future", "reuse", "should be remembered", "long-term", "长期", "记住", "复用", "有用"]
    return any(word in lowered for word in durable_words)
