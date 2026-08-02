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
from .interventions import (
    HALFMNIST_HALF_SWAP,
    HALFMNIST_PATCH_CONFLICT,
    HALFMNIST_PATCH_NEUTRAL,
    HALFMNIST_PATCH_REMOVED,
    HALFMNIST_PATCH_SHUFFLED,
    HalfMNISTLabelPatchDataset,
    PredictionIntervention,
    align_identity_concept_probabilities,
    align_swapped_concept_probabilities,
    apply_halfmnist_label_patch,
    contradict_halfmnist_label_patch,
    get_intervention,
    neutralize_halfmnist_label_patch,
    remove_halfmnist_label_patch,
    shuffle_halfmnist_label_patch,
    swap_halfmnist_image_halves,
)
from .experiment import (
    MetaBEARSCalibration,
    MetaBEARSExperimentResult,
    build_calibrated_report,
    calibrate_metabears,
    run_metabears_experiment,
    shortcut_proxy_labels,
)
from .report import MetaCognitiveReport, build_meta_cognitive_report
from .thresholds import ThresholdSelection, select_review_threshold

__all__ = [
    "ConsistencyResult",
    "EnsemblePredictions",
    "MetaBEARSCalibration",
    "MetaBEARSExperimentResult",
    "MetaCognitiveReport",
    "PredictionIntervention",
    "ThresholdSelection",
    "HALFMNIST_HALF_SWAP",
    "HALFMNIST_PATCH_CONFLICT",
    "HALFMNIST_PATCH_NEUTRAL",
    "HALFMNIST_PATCH_REMOVED",
    "HALFMNIST_PATCH_SHUFFLED",
    "HalfMNISTLabelPatchDataset",
    "align_identity_concept_probabilities",
    "align_swapped_concept_probabilities",
    "apply_halfmnist_label_patch",
    "build_meta_cognitive_report",
    "build_calibrated_report",
    "calibrate_metabears",
    "collect_ensemble_predictions",
    "contradict_halfmnist_label_patch",
    "ensemble_leave_one_out_reference_distances",
    "ensemble_nearest_reference_distances",
    "familiarity_from_reference",
    "get_intervention",
    "neutralize_halfmnist_label_patch",
    "remove_halfmnist_label_patch",
    "probe_concept_consistency",
    "run_metabears_experiment",
    "select_review_threshold",
    "shortcut_proxy_labels",
    "shuffle_halfmnist_label_patch",
    "swap_halfmnist_image_halves",
]
