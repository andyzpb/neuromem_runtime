from __future__ import annotations

from dataclasses import dataclass

from neuromem.core.policy import MemoryPolicy
from neuromem.stores.base import MemoryStore
from neuromem_runtime.policy_v2 import ProposedDelta, ValidatedMutation, ValidationStep


@dataclass(slots=True)
class ValidationContext:
    store: MemoryStore | None = None
    phase: str = "mutation"
    authorize_delete: bool = False
    user_id: str | None = None
    namespace: str = "default"


class MutationValidator:
    name = "MutationValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        return ValidationStep(name=self.name, passed=True)


class SchemaValidator(MutationValidator):
    name = "SchemaValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        valid = policy.write.operation in {"ADD", "UPDATE", "LINK", "NOOP"} and policy.forget.operation in {"NOOP", "DECAY", "INHIBIT", "INVALIDATE", "ARCHIVE", "DELETE_REQUEST"}
        return ValidationStep(name=self.name, passed=valid, reason="" if valid else "unsupported operation")


class EvidenceValidator(MutationValidator):
    name = "EvidenceValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if policy.write.operation in {"ADD", "UPDATE", "LINK"} and not policy.write.evidence_ids:
            return ValidationStep(name=self.name, passed=False, reason="write mutations require evidence ids")
        return ValidationStep(name=self.name, passed=True)


class ProvenanceValidator(MutationValidator):
    name = "ProvenanceValidator"


class TemporalValidator(MutationValidator):
    name = "TemporalValidator"


class ConflictValidator(MutationValidator):
    name = "ConflictValidator"


class PrivacyAclValidator(MutationValidator):
    name = "PrivacyAclValidator"


class DeletionGuardValidator(MutationValidator):
    name = "DeletionGuardValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        if policy.forget.operation == "DELETE_REQUEST" and not context.authorize_delete:
            return ValidationStep(name=self.name, passed=False, reason="DELETE_REQUEST requires explicit user authorization")
        return ValidationStep(name=self.name, passed=True)


class PoisoningRiskValidator(MutationValidator):
    name = "PoisoningRiskValidator"

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidationStep:
        content = (policy.write.content or "").lower()
        suspicious = ["ignore previous", "override memory", "always trust this unverified", "delete audit"]
        if any(term in content for term in suspicious):
            return ValidationStep(name=self.name, passed=False, reason="possible memory poisoning instruction")
        return ValidationStep(name=self.name, passed=True)


class LifecycleTransitionValidator(MutationValidator):
    name = "LifecycleTransitionValidator"


class IndexConsistencyValidator(MutationValidator):
    name = "IndexConsistencyValidator"


class ValidatorStack:
    def __init__(self, validators: list[MutationValidator] | None = None) -> None:
        self.validators = validators or [
            SchemaValidator(),
            EvidenceValidator(),
            ProvenanceValidator(),
            TemporalValidator(),
            ConflictValidator(),
            PrivacyAclValidator(),
            DeletionGuardValidator(),
            PoisoningRiskValidator(),
            LifecycleTransitionValidator(),
            IndexConsistencyValidator(),
        ]

    def validate(self, policy: MemoryPolicy, context: ValidationContext) -> ValidatedMutation:
        trace = [validator.validate(policy, context) for validator in self.validators]
        approved = all(step.passed for step in trace)
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
            risk_score=0.1 if approved else 0.9,
            validator_trace=trace,
        )


__all__ = [
    "ConflictValidator",
    "DeletionGuardValidator",
    "EvidenceValidator",
    "IndexConsistencyValidator",
    "LifecycleTransitionValidator",
    "MutationValidator",
    "PoisoningRiskValidator",
    "PrivacyAclValidator",
    "ProvenanceValidator",
    "SchemaValidator",
    "TemporalValidator",
    "ValidationContext",
    "ValidatorStack",
]
