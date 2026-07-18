"""MetaBEARS phase-II diagnostic components.

The package operates on NumPy arrays so it can consume predictions collected
from BEARS ensembles, Monte-Carlo dropout runs, or compatible NeSy models
without coupling the diagnostic logic to a training framework.
"""

from .consistency import ConsistencyResult, probe_concept_consistency
from .familiarity import familiarity_from_reference
from .report import MetaCognitiveReport, build_meta_cognitive_report

__all__ = [
    "ConsistencyResult",
    "MetaCognitiveReport",
    "build_meta_cognitive_report",
    "familiarity_from_reference",
    "probe_concept_consistency",
]
