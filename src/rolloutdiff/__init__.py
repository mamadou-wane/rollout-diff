"""Deterministic, label-free rollout analysis."""

from .adapters import from_table, load_trl, load_verl
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
from .pipeline import AnalysisResult, analyze_rollouts
from .prompt_mix import PromptMixEvidence, prompt_mix
from .ranking import rank_events
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
    "aggregate",
    "analyze_rollouts",
    "build_report",
    "detect_retrospective",
    "detect_sequential",
    "event_window_corroboration",
    "extract",
    "featurize",
    "from_table",
    "group_events",
    "load_trl",
    "load_verl",
    "persistence",
    "prompt_mix",
    "rank_events",
    "render_report",
    "resolve_exemplars",
    "reward_association",
    "select_exemplars",
    "split_half_replication",
    "validate_rollouts",
]
