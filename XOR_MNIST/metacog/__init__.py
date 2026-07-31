"""MetaBEARS phase-II diagnostic components.

The package operates on NumPy arrays so it can consume predictions collected
from BEARS ensembles, Monte-Carlo dropout runs, or compatible NeSy models
without coupling the diagnostic logic to a training framework.
"""

from .consistency import ConsistencyResult, probe_concept_consistency
from .familiarity import familiarity_from_reference
from .integration import (
    EnsemblePredictions,
    collect_ensemble_predictions,
    ensemble_leave_one_out_reference_distances,
    ensemble_nearest_reference_distances,
)
from .report import MetaCognitiveReport, build_meta_cognitive_report
from .thresholds import ThresholdSelection, select_review_threshold

__all__ = [
    "ConsistencyResult",
    "EnsemblePredictions",
    "MetaCognitiveReport",
    "ThresholdSelection",
    "build_meta_cognitive_report",
    "collect_ensemble_predictions",
    "ensemble_leave_one_out_reference_distances",
    "ensemble_nearest_reference_distances",
    "familiarity_from_reference",
    "probe_concept_consistency",
    "select_review_threshold",
]
