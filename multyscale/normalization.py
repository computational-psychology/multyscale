from collections.abc import Sequence

# Third party imports
import numpy as np

# Local application imports
from . import filters

# Generalized (F)(L)ODOG normalization


def normalizers(multioutput: np.ndarray, normalization_weights: np.ndarray) -> np.ndarray:
    """Construct all normalizers: weighted combination of all fitler outputs, for each filter

    Parameters
    ----------
    multioutput : numpy.ndarray
        all (O, S, X, Y) filter outputs
    normalization_weights : numpy.ndarray
        full tensor (O, S, O, S) of normalization weights

    Returns
    -------
    numpy.ndarray
        all (O, S, X, Y) normalizers:
        a single 2D (X,Y) weighted combination of all filter outputs
        per (O, S) filter-to-normalize
    """

    # Create normalizing images from weighted combination of filter outputs
    norms: np.ndarray = np.ndarray(shape=multioutput.shape)
    for o, s in np.ndindex(multioutput.shape[:2]):
        weights = normalization_weights[o, s]

        # Tensor dot: multiply filters_output by weights, then sum over axes [0,1]
        normalizer = np.tensordot(multioutput, weights, axes=([0, 1], [0, 1]))
        # Accumulate
        norms[o, s, ...] = normalizer

    return norms


def norm_coeff(normalizer: np.ndarray, spatial_kernel: np.ndarray) -> np.ndarray:
    """Construct normalization coefficient: denominator for divisive normalization

    Parameters
    ----------
    normalizer : numpy.ndarray
        single 2D normalizer image; weighted combination of all filter outputs
    spatial_kernel : numpy.ndarray
        single kernel to spatially average (2D; over x,y) the normalizer

    Returns
    -------
    numpy.ndarray
        single normalization coefficient: denominator for divisive normalization
    """
    norm = normalizer**2
    spatial_average = filters.apply(norm, spatial_kernel, padval=0)
    coeff = np.sqrt(spatial_average)
    return coeff


def divisive_normalization(filter_output: np.ndarray, norm_coeff: np.ndarray) -> np.ndarray:
    """Apply divisive normalization to a single filter output

    Parameters
    ----------
    filter_output : np.ndarray
        output from a single filter, 2D of shape (x,y)
    norm_coeff : np.ndarray
        single normalization coefficient: denominator for divisive normalization

    Returns
    -------
    np.ndarray
        normalized filter output, 2D of shape (x, y)
    """
    return filter_output / norm_coeff


def spatial_kernel_globalmean(shape: Sequence[int]) -> np.ndarray:
    """Create spatial averaging kernel, calculating the global image mean

    Parameters
    ----------
    shape : (int, int)
        shape in pixels (height, width) of image to average

    Returns
    -------
    numpy.ndarray
        spatial averaging kernel
    """
    return np.ones([dim * 2 for dim in shape]) / np.prod(shape)


def spatial_kernel_gaussian(x: np.ndarray, y: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    """Create spatial averaging kernel, in a 2D Gaussian shape

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates of each pixel
    y : numpy.ndarray
        y coordinates of each pixel
    sigmas : (float, float)
        standard deviation along each axis (2-vector) or both axes (scalar)

    Returns
    -------
    numpy.ndarray
        spatial averaging kernel
    """
    kernel = filters.gaussian2d(x, y, sigmas)
    kernel /= kernel.sum()
    return kernel


def create_normalization_weights(
    n_orientations: int,
    n_scales: int,
    scale_norm_weights: np.ndarray,
    orientation_norm_weights: np.ndarray,
) -> np.ndarray:
    """Combine normalization weights across dimensions into single tensor of weights

    Each filter can, in theory, normalized by every (other) filter,
    but the weight for how much every filter normalizes, can vary.
    For each filter, create a tensor with weights for each other filter.

    Parameters
    ----------
    n_orientations : int
        number of orientations to weight
    n_scales : int
        number of spatial scales/frequencies to weight
    scale_norm_weights : numpy.ndarray
        matrix of normalization weights across scales, of shape (n_scales, n_scales)
    orientation_norm_weights : numpy.ndarray
        matrix of normalization weights across orientations, of shape (n_orientations, n_orientations)

    Returns
    -------
    numpy.ndarray
        tensor of combined normalization weights,
        of shape (n_orientations, n_scales, n_orientations, n_scales) --
        for each filter $(o', s')$, a matrix of $(O, S)$ normalization weights
    """
    normalization_weights: np.ndarray = np.ndarray(
        shape=(n_orientations, n_scales, n_orientations, n_scales)
    )
    for o_prime, s_prime in np.ndindex(n_orientations, n_scales):
        normalization_weights[o_prime, s_prime, ...] = np.outer(
            orientation_norm_weights[o_prime], scale_norm_weights[s_prime]
        )
    return normalization_weights


def scale_norm_weights_equal(n_scales: int) -> np.ndarray:
    """Equal weighting for all scales

    All filters $(o=o', s)$ get some weight;
    in (L)ODOG, that weight is $1/len(S)$,

    Parameters
    ----------
    n_scales : int
        number of scales to create weights for

    Returns
    -------
    numpy.ndarray
        matrix of normalization weights, of shape (n_scales, n_scales)
    """
    return np.ones((n_scales, n_scales))


def scale_norm_weights_gaussian(n_scales: int, sdmix: float) -> np.ndarray:
    """Gaussian weighting for all scales

    All filters $(o=o', s)$ get some weight;
    in FLODOG, that weight is Gaussian dependent on $s-s'$

    Parameters
    ----------
    n_scales : int
        number of scales to create weights for
    sdmix : float
        standard deviation (SD) of Gaussian

    Returns
    -------
    numpy.ndarray
        matrix of normalization weights, of shape (n_scales, n_scales)
    """
    scale_norm_weights: np.ndarray = np.ndarray(shape=(n_scales, n_scales))
    for s in range(n_scales):
        rel_i = np.asarray(range(n_scales)) - s

        # Gaussian weights, based on relative index
        weights = np.exp(-(rel_i**2) / (2 * sdmix**2)) / (sdmix * np.sqrt(2 * np.pi))
        weights /= weights.sum()
        scale_norm_weights[s, :] = weights

    return scale_norm_weights


def orientation_norm_weights(n_orientations: int) -> np.ndarray:
    """Self-normalization weight for each orientation

    Each filter is normalized by other filters of same orientation,
    but not by filters at other orientations;
    so for filter $(o',s')$, all filters $(o=o', ...)$ get weight $1$, else $0$.

    Parameters
    ----------
    n_orientations : int
        number of orientations to create weights for

    Returns
    -------
    numpy.ndarray
        matrix of normalization weights, of shape (n_orientations, n_orientations)
    """
    orientation_norm_weights = np.eye(n_orientations)
    orientation_norm_weights /= orientation_norm_weights.sum(0)
    return orientation_norm_weights
