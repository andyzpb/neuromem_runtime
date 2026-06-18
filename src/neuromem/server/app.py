from __future__ import annotations

from neuromem.core.runtime import NeuroMemRuntime


def create_runtime() -> NeuroMemRuntime:
    return NeuroMemRuntime(agent_id="demo-agent", namespace="default")

