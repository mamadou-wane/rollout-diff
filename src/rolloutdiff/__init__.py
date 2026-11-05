"""Deterministic, label-free rollout analysis."""

from .adapters import from_table, load_trl, load_verl
from .comparison import compare_rollouts, resolve_comparison_citations
from .comparison_report import (
    comparison_json,
    load_comparison_report,
    render_comparison_report,
    validate_comparison_report,
)
from .corroboration import EventWindowEvidence, event_window_corroboration
from .detect import DetectionResult, detect_sequential
from .events import EventGrouping, group_events
from .evidence import (
    PersistenceEvidence,
    RewardAssociation,
    persistence,
    reward_association,
)
from .exemplars import select_exemplars
from .features import FeatureTables, RolloutFeatures, extract, featurize
from .html_report import render_report
from .inspection import resolve_exemplars
from .investigation import (
    approve_aria_intervention,
    compare_aria_intervention,
    execute_aria_intervention,
    prepare_aria_intervention,
    render_aria_comparison,
)
from .pipeline import AnalysisResult, analyze_rollouts
from .prompt_mix import PromptMixEvidence, prompt_mix
from .ranking import rank_events
from .replay import SandboxConfig, replay_aria, serialize_replay
from .replication import ReplicationEvidence, split_half_replication
from .report import build_report
from .retrospective import RetrospectiveResult, detect_retrospective
from .schema import Rollout, validate_rollouts
from .series import Aggregation, aggregate

__all__ = [
    "Aggregation",
    "AnalysisResult",
    "DetectionResult",
    "EventGrouping",
    "EventWindowEvidence",
    "FeatureTables",
    "PersistenceEvidence",
    "PromptMixEvidence",
    "ReplicationEvidence",
    "RetrospectiveResult",
    "RewardAssociation",
    "Rollout",
    "RolloutFeatures",
    "SandboxConfig",
    "aggregate",
    "analyze_rollouts",
    "approve_aria_intervention",
    "build_report",
    "compare_aria_intervention",
    "compare_rollouts",
    "comparison_json",
    "detect_retrospective",
    "detect_sequential",
    "event_window_corroboration",
    "execute_aria_intervention",
    "extract",
    "featurize",
    "from_table",
    "group_events",
    "load_comparison_report",
    "load_trl",
    "load_verl",
    "persistence",
    "prepare_aria_intervention",
    "prompt_mix",
    "rank_events",
    "render_aria_comparison",
    "render_comparison_report",
    "render_report",
    "replay_aria",
    "resolve_comparison_citations",
    "resolve_exemplars",
    "reward_association",
    "select_exemplars",
    "serialize_replay",
    "split_half_replication",
    "validate_comparison_report",
    "validate_rollouts",
]
