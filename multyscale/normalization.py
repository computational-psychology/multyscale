from collections.abc import Sequence

# Third party imports
import numpy as np

# Local application imports
from . import filters

# Generalized (F)(L)ODOG normalization


def norm_coeffs(multioutput: np.ndarray, normalization_weights: np.ndarray) -> np.ndarray:
    """Construct all normalizing coefficients: weighted combination of all filter outputs, for each filter

    The multi(dimensional)output consists of all (O, S) filter outputs,
    where each is of dimension (Y, X).
    Thus, the multioutput has dimension (O, S, Y, X).

    The output will be one normalizing coefficient, a matrix of (Y, X) pixels,
    for each filter-to-normalize f'(o', s').
    Thus, the norm_coeffs output also has dimension (O', S', Y, X).

    To create these, for each filter-to-normalize f'(o', s'),
    there should be (O, S) weights provide:
    one for how each filter is weighted in this normalizing coefficient.
    Thus, the normalization weights are of dimension (O', S', O, S):
    for each (o',s') filter-to-normalize, there are (O, S) weights.

    Parameters
    ----------
    multioutput : numpy.ndarray
        all (O, S, Y, X) filter outputs
    normalization_weights : numpy.ndarray
        full tensor (O', S', O, S) of normalization weights

    Returns
    -------
    numpy.ndarray
        all (O', S', Y, X) normalizing coefficients:
        a single 2D (Y, X) weighted combination of all filter outputs
        per (O', S') filter-to-normalize
    """

    # Create all normalizing coefficients from weighted combination of filter outputs
    # This is done as a single tensor dot-product operation.
    # Multiply the (O', S', O, S) weights, by the (O, S, Y, X) filter outputs,
    # then sum over the (O, S) dimensions.
    # In Einstein summation notation (and q := o', r := s'), that is:
    #
    # norms = np.einsum('qros, osyx -> qryx', normalization_weights, multioutput)
    #
    # which in tensordot syntax is:
    norms = np.tensordot(normalization_weights, multioutput, axes=([2, 3], [0, 1]))

    return norms


def norm_energy(norm_coeff: np.ndarray, spatial_kernel: np.ndarray, eps=0.0) -> np.ndarray:
    """Convert normalizing coefficient to energy (denominator for divisive normalization)

    Parameters
    ----------
    norm_coeff : numpy.ndarray
        single 2D normalizing coefficient; weighted combination of all filter outputs
    spatial_kernel : numpy.ndarray
        single kernel to spatially average (2D; over x,y) the normalizing coefficient
    eps : float, optional
        precision offset, used to avoid square-root of negative numbers, by default 0.0

    Returns
    -------
    numpy.ndarray
        single normalizing energy: denominator for divisive normalization
    """
    norm = norm_coeff**2
    spatial_average = filters.apply(norm, spatial_kernel, padval=0)
    energy = np.sqrt(spatial_average + eps)
    return energy


def divisive_normalization(
    filter_output: np.ndarray, norm_coeff: np.ndarray, eps=0.0
) -> np.ndarray:
    """Apply divisive normalization to a single filter output

    Parameters
    ----------
    filter_output : np.ndarray
        output from either a single filter, 2D of shape (Y, X),
        or a whole filterbank, N-Dimensional of shape (M, ..., N, Y, X)
    norm_coeff : np.ndarray
        normalizing coefficient(s): denominator for divisive normalization.
        Must be of same shape as filter_output
    eps : float, optional
        precision offset, used to avoid DivideByZero errors, by default 0.0

    Returns
    -------
    np.ndarray
        normalized filter output, 2D of shape (y, x)
    """
    return filter_output / (norm_coeff + eps)


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
    kernel = np.ones([dim * 2 for dim in shape])
    kernel /= kernel.sum()
    kernel *= 4
    return kernel


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
