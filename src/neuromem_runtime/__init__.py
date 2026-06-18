from __future__ import annotations

from neuromem.core.policy import MemoryPolicy, MemoryTrace

from neuromem_runtime.runtime import MemoryRuntime
from neuromem_runtime.deltas import ExecutionDeltaPlan, GraphDelta, IndexDelta, LifecycleDelta, MemoryDelta, MemorySnapshot, MutationExecutionResult
from neuromem_runtime.ledger import ExperienceEvent, LedgerEvent, MemoryLedger
from neuromem_runtime.lifecycle import LifecycleStateMachine
from neuromem_runtime.policy_v2 import GraphDeltaProposal, MemoryPolicyV2, ValidatedMutation
from neuromem_runtime.plasticity import PlasticityEngine
from neuromem_runtime.retrieval import (
    ActivationResult,
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    EntityAliasResolver,
    HyDEProvider,
    LocalVectorIndex,
    MemoryCard,
    QueryPlanV2,
    QueryRewriteProvider,
    RerankProvider,
    RetrievalCandidate,
    RetrievalConfig,
    RetrievalLedgerRecord,
    RetrievalTraceMetadata,
    StaticEntityAliasResolver,
    VectorIndex,
)
from neuromem_runtime.semantic_graph import (
    DeterministicRelationProposer,
    GraphBuildContext,
    GraphCandidateGenerator,
    GraphDeltaValidator,
    GraphMutationCommitter,
    GraphProposalProvider,
    GraphRelationCandidate,
)
from neuromem_runtime.sleep import SleepPlanner, SleepReport
from neuromem_runtime.types import EvidenceBundle, MemoryContext, MemoryEvent, MemoryQuery, MemoryTransaction, RuntimeConfig
from neuromem_runtime.validators import ValidatorStack
from neuromem_runtime.providers import DeepSeekPolicyProvider, DeterministicPolicyProvider, OpenAICompatiblePolicyProvider, PolicyProvider

__version__ = "0.2.0"

__all__ = [
    "MemoryRuntime",
    "RuntimeConfig",
    "MemoryEvent",
    "MemoryQuery",
    "MemoryContext",
    "EvidenceBundle",
    "MemoryPolicy",
    "MemoryTransaction",
    "MemoryTrace",
    "ExperienceEvent",
    "MemoryPolicyV2",
    "GraphDeltaProposal",
    "ValidatedMutation",
    "ExecutionDeltaPlan",
    "MemoryDelta",
    "GraphDelta",
    "LifecycleDelta",
    "IndexDelta",
    "MemorySnapshot",
    "MutationExecutionResult",
    "LedgerEvent",
    "MemoryLedger",
    "LifecycleStateMachine",
    "ValidatorStack",
    "EmbeddingProvider",
    "EntityAliasResolver",
    "HyDEProvider",
    "LocalVectorIndex",
    "VectorIndex",
    "QueryRewriteProvider",
    "StaticEntityAliasResolver",
    "RetrievalConfig",
    "QueryPlanV2",
    "MemoryCard",
    "RetrievalCandidate",
    "ActivationResult",
    "RetrievalLedgerRecord",
    "RerankProvider",
    "RetrievalTraceMetadata",
    "DeterministicEmbeddingProvider",
    "GraphBuildContext",
    "GraphCandidateGenerator",
    "GraphRelationCandidate",
    "GraphProposalProvider",
    "DeterministicRelationProposer",
    "GraphDeltaValidator",
    "GraphMutationCommitter",
    "PlasticityEngine",
    "SleepPlanner",
    "SleepReport",
    "PolicyProvider",
    "DeterministicPolicyProvider",
    "OpenAICompatiblePolicyProvider",
    "DeepSeekPolicyProvider",
]
