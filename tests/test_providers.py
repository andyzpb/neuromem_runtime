from __future__ import annotations

import json

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
