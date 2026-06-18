from __future__ import annotations

import json

from neuromem_runtime.policy_v2 import MemoryPolicyV2
from neuromem_runtime.providers import DeterministicPolicyProvider, extract_policy_payload, memory_policy_from_payload, normalize_policy_payload


def test_extract_policy_payload_from_wrapped_json() -> None:
    payload = {
        "retrieval": {"enabled": False, "query": ""},
        "write": {"operation": "NOOP"},
        "forget": {"operation": "NOOP"},
        "consolidation": {"enabled": False},
        "reason": "ok",
    }
    extracted = extract_policy_payload(f"prefix {json.dumps(payload)} suffix")
    policy = memory_policy_from_payload(extracted)
    assert policy.reason == "ok"
    assert policy.source == "small_llm"


def test_deterministic_provider_write_policy() -> None:
    provider = DeterministicPolicyProvider()
    policy = provider.propose({"phase": "after_step", "content": "Remember this.", "evidence": "trace-1"})
    assert policy.write.operation == "ADD"
    assert policy.write.evidence_ids == ["trace-1"]


def test_memory_policy_v2_payload_returns_v2_policy() -> None:
    payload = {
        "policy_id": "policy-test",
        "proposal_source": "small_llm",
        "intent": "add",
        "risk_level": "low",
        "evidence_chain": [{"event_id": "evt-1", "source": "test"}],
        "target_selector": {},
        "proposed_deltas": [{"operation": "ADD", "value": {"content": "Remember this.", "memory_type": "semantic"}, "reason": "test"}],
        "safety_annotations": {},
        "temporal_scope": "all_valid",
        "retention_policy": "keep_for_audit",
        "rollback_plan": "rollback",
    }
    extracted = extract_policy_payload(f"prefix {json.dumps(payload)} suffix")
    policy = memory_policy_from_payload(extracted)
    assert isinstance(policy, MemoryPolicyV2)
    assert policy.intent == "add"


def test_memory_policy_v2_normalizes_temporal_scope_object() -> None:
    payload = {
        "policy_id": "policy-test",
        "proposal_source": "small_llm",
        "intent": "add",
        "risk_level": "low",
        "evidence_chain": [{"event_id": "evt-1", "source": "test"}],
        "target_selector": {},
        "proposed_deltas": [{"operation": "ADD", "value": {"content": "User visited Ginza.", "memory_type": "episodic"}, "reason": "test"}],
        "safety_annotations": {},
        "write_gate": {
            "decision": "commit",
            "durability_horizon": "long_term",
            "commitment_level": "durable_memory",
            "basis": "current_user_message",
            "signals": ["future_utility"],
            "rationale": "The user provided a travel memory.",
        },
        "temporal_scope": {"from": None, "to": None},
        "retention_policy": "keep_for_audit",
        "rollback_plan": "rollback",
    }

    normalized = normalize_policy_payload({"after_step": payload})
    policy = memory_policy_from_payload({"after_step": payload})

    assert normalized["temporal_scope"] == "all_valid"
    assert isinstance(policy, MemoryPolicyV2)
    assert policy.temporal_scope == "all_valid"


def test_memory_policy_v2_normalizes_target_selector_extra_keys() -> None:
    payload = {
        "policy_id": "policy-test",
        "proposal_source": "small_llm",
        "intent": "update",
        "risk_level": "low",
        "evidence_chain": [{"event_id": "evt-1", "source": "test"}],
        "target_selector": {"memory_ids": "mem-1", "scope": "all_valid"},
        "proposed_deltas": [{"operation": "UPDATE", "target_memory_id": "mem-1", "value": {"content": "Tom has the sushi habit."}, "reason": "correction"}],
        "safety_annotations": {},
        "write_gate": {
            "decision": "commit",
            "durability_horizon": "long_term",
            "commitment_level": "durable_memory",
            "basis": "current_user_message",
            "signals": ["correction"],
            "rationale": "The user corrected a prior memory.",
        },
        "temporal_scope": "all_valid",
        "retention_policy": "keep_for_audit",
        "rollback_plan": "rollback",
    }

    normalized = normalize_policy_payload({"after_step": payload})
    policy = memory_policy_from_payload({"after_step": payload})

    assert normalized["target_selector"] == {"memory_ids": ["mem-1"]}
    assert isinstance(policy, MemoryPolicyV2)
    assert policy.target_selector.memory_ids == ["mem-1"]
