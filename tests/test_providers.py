from __future__ import annotations

import json

from neuromem_runtime.policy_v2 import MemoryPolicyV2
from neuromem_runtime.providers import DeterministicPolicyProvider, extract_policy_payload, memory_policy_from_payload


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
