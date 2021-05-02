# Third party imports
import numpy as np

# Local application imports
from . import filters


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


def scale_norm_weights_gaussian(n_scales, sdmix):
    scale_norm_weights = np.ndarray((n_scales, n_scales))
    for s in range(n_scales):
        rel_i = s - np.asarray(range(n_scales))

        # Gaussian weights, based on relative index
        scale_norm_weights[s, ...] = np.exp(-(rel_i ** 2) / 2 * sdmix ** 2) / (
            sdmix * np.sqrt(2 * np.pi)
        )
    return scale_norm_weights


def orientation_norm_weights(n_orientations):
    orientation_norm_weights = np.eye(n_orientations)
    return orientation_norm_weights


def normalizers(filters_output, normalization_weights):
    # Create normalizing images from weighted combination of filter outputs
    normalizers = np.ndarray(filters_output.shape)

    for o, s in np.ndindex(filters_output.shape[:2]):
        weights = normalization_weights[o, s]

        # Tensor dot: multiply filters_output by weights, then sum over axes [0,1]
        normalizer = np.tensordot(filters_output, weights, axes=([0, 1], [0, 1]))

        # Normalize normalizer...
        area = weights.sum()
        normalizer = normalizer / area

        # Accumulate
        normalizers[o, s, ...] = normalizer

    return normalizers


def spatial_avg_windows_globalmean(normalizers):
    filts = np.ndarray(normalizers.shape)
    for o, s in np.ndindex(filts.shape[:2]):
        filts[o, s] = filters.global_avg(normalizers.shape[2:])
    return filts


def nomalizers_to_RMS(normalizers, spatial_avg_filters):
    normalizers_RMS = normalizers.copy()
    normalizers_RMS = np.square(normalizers_RMS)
    normalizers_RMS = spatial_avg_filters * normalizers_RMS
    normalizers_RMS = normalizers_RMS.sum(axis=(-1, -2))
    normalizers_RMS = np.sqrt(normalizers_RMS)
    return normalizers_RMS
