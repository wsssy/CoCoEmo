"""Linear-separability probing of emotion representations across layers/operators.

Used both for the "where to steer" analysis (Section 2 of the paper) and to
supply geometry helpers (``HeadGeometry``) to the steering module.
"""

from ._probe import (
    HeadGeometry,
    LayerDiscriminabilityMetrics,
    HeadDiscriminabilityMetrics,
    compute_discriminability_for_steering,
    compute_layer_discriminability,
    print_discriminability_report,
    find_best_layers_for_steering,
)

__all__ = [
    "HeadGeometry",
    "LayerDiscriminabilityMetrics",
    "HeadDiscriminabilityMetrics",
    "compute_discriminability_for_steering",
    "compute_layer_discriminability",
    "print_discriminability_report",
    "find_best_layers_for_steering",
]
