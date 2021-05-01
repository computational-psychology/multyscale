# Third party imports
import numpy as np

# Local application imports


# DOG_BM1997 normalization

# ODOG_BM1999 normalization
# Each filter is normalized by other filters of same orientation,
# but not by filters at other orientations
# Different scales are weighted equally
# so for filter (o,s), all filters (O!=o, :) get weight 0,
# and all filters (O=o, S) get weight 1/len(S)
def create_normalization_weights(
    n_orientations, n_scales, scale_norm_weights, orientation_norm_weights
):
    # Each filter can, in theory, normalized by every (other) filter
    # But the weight for how much every filter normalizes, can vary
    # For each filter, create a tensor with weights for each other filter
    normalization_weights = np.ndarray(
        shape=(n_orientations, n_scales, n_orientations, n_scales)
    )

    for o, s in np.ndindex(n_orientations, n_scales):
        normalization_weights[o, s, ...] = np.outer(
            orientation_norm_weights[o], scale_norm_weights[s]
        )
    return normalization_weights


def scale_norm_weights_equal(n_scales):
    return np.ones((n_scales, n_scales)) / n_scales


def orientation_norm_weights(n_orientations):
    orientation_norm_weights = np.eye(n_orientations)
    return orientation_norm_weights
