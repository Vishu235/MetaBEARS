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
    shuffled_patch_assignment_metrics,
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
from .posthoc import (
    FusionRunAnalysis,
    PosthocRunAnalysis,
    ValidationFusionModel,
    detector_scores_and_targets,
    evaluate_detector_arrays,
    evaluate_fusion_arrays,
    evaluate_fusion_result_directory,
    evaluate_result_directory,
    fit_validation_fusion,
    fit_validation_fusion_from_result_directory,
    precision_recall_curve,
    risk_coverage_curve,
)
from .thresholds import ThresholdSelection, select_review_threshold

__all__ = [
    "ConsistencyResult",
    "EnsemblePredictions",
    "MetaBEARSCalibration",
    "MetaBEARSExperimentResult",
    "MetaCognitiveReport",
    "PredictionIntervention",
    "FusionRunAnalysis",
    "PosthocRunAnalysis",
    "ValidationFusionModel",
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
    "detector_scores_and_targets",
    "evaluate_detector_arrays",
    "evaluate_fusion_arrays",
    "evaluate_fusion_result_directory",
    "evaluate_result_directory",
    "fit_validation_fusion",
    "fit_validation_fusion_from_result_directory",
    "familiarity_from_reference",
    "get_intervention",
    "neutralize_halfmnist_label_patch",
    "remove_halfmnist_label_patch",
    "probe_concept_consistency",
    "precision_recall_curve",
    "risk_coverage_curve",
    "run_metabears_experiment",
    "select_review_threshold",
    "shortcut_proxy_labels",
    "shuffle_halfmnist_label_patch",
    "shuffled_patch_assignment_metrics",
    "swap_halfmnist_image_halves",
]
