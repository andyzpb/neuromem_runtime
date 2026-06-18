from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalPlan:
    mode: str = "hybrid"
    memory_types: list[str] = field(default_factory=list)
    exclude_maturity: list[str] = field(default_factory=lambda: ["deleted", "obsolete", "inhibited"])
    context_budget_tokens: int = 1500
    use_entities: bool = True
    use_temporal: bool = True
    use_graph: bool = True


@dataclass(slots=True)
class WriteDecision:
    destination: str
    reason: str


class PFCController:
    def plan_retrieval(self, query: str, state: dict[str, object] | None = None, budget_tokens: int = 1500) -> RetrievalPlan:
        text = f"{query} {state or ''}".lower()
        memory_types: list[str] = []
        if any(term in text for term in ["bug", "failure", "error", "regression"]):
            memory_types.extend(["episodic", "procedural"])
        if any(term in text for term in ["preference", "current", "fact", "command", "test", "testing"]):
            memory_types.extend(["semantic", "preference", "procedural", "episodic"])
        return RetrievalPlan(memory_types=list(dict.fromkeys(memory_types)), context_budget_tokens=budget_tokens)

    def decide_write(self, salience_score: float, event: dict[str, object]) -> WriteDecision:
        event_type = str(event.get("type", ""))
        if event_type in {"failure", "task_result"}:
            return WriteDecision("episodic", "agent outcome belongs in hippocampal memory")
        if event_type in {"fact", "user_preference"}:
            return WriteDecision("long_term", "stable fact or preference belongs in neocortical memory")
        if event_type == "rule":
            return WriteDecision("long_term", "rule belongs in procedural memory")
        if salience_score >= 0.75:
            return WriteDecision("long_term", "high salience event")
        if salience_score >= 0.45:
            return WriteDecision("episodic", "medium salience event")
        if salience_score >= 0.25:
            return WriteDecision("working", "low salience event")
        return WriteDecision("discard", "below write threshold")
