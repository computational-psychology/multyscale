"""End-to-End multiscale spatial filtering models

This module implements several existing multiscale spatial filtering models.
The models here are primarily used to model brightness perception.

Each model-type is implement as a (separate) class;
the parameters, attributes and methods available differ per class.
All models require a filter `shape` in pixels (X, Y),
and a `visextent` in degrees of visual angle (left, right, top, bottom)
All models have a method `.apply()`,
which takes in a (2D) image and returns the final model output (also 2D image).

Currently implemented are:
- Difference-of-Gaussian (Blakeslee & McCourt, 1997), `DOG_BM1997`
- Oriented Difference-of-Gaussian (ODOG; Blakeslee & McCourt, 1999; Robinson, Hammon & de Sa, 2007)
    `ODOG_RHS2007`
- LODOG (Robinson, Hammon & de Sa, 2007) `LODOG_RHS2007`
- FLODOG (Robinson, Hammon & de Sa, 2007) `FLODOG_RHS2007`

"""

from typing import Sequence

# Third party imports
import numpy as np

# Local application imports
from . import filterbanks, normalization


class DOG_BM1997:
    """Difference-of-Gaussian model, after Blakeslee & McCourt (1997)

    This model uses 7 unoriented (isotropic) Difference-of-Gaussian filters,
    weighted according to an (approximate) contrast sensitivity function.

    Parameters
    ----------
    shape : array-like[int]
        pixel size (X, Y) of filters
    visextent : array-like[float]
        extent in degrees visual angle, (left, right, top, bottom)

    See also
    --------
    multyscale.filterbanks.BM1997

    """

    def __init__(self, shape: Sequence[int], visextent: Sequence[float]):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbanks.BM1997(shape, visextent)

        self.center_sigmas = [sigma[0] for sigma in self.bank.sigmas]
        self.weights_slope = 0.1
        self.scale_weights = filterbanks.scale_weights(self.center_sigmas, self.weights_slope)

    def weight_outputs(self, filters_output: np.ndarray) -> np.ndarray:
        """Weight filter outputs according to spatial scale of filter

        Uses self.scale_weights for weighting

        Parameters
        ----------
        filters_output : numpy.ndarray
            output of whole filterbank, of shape (S, Y, X)

        Returns
        -------
        numpy.ndarray
            weighted output of whole filterbank, same shape is filters_output
        """
        return filterbanks.weight_multiscale_outputs(filters_output, self.scale_weights)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply model to given image

        Parameters
        ----------
        image : numpy.ndarray
            image, 2D in shape (Y, X)
        eps : float, optional
            precision offset, used to avoid floating point errors, by default 0.0

        Returns
        -------
        numpy.ndarray
            matrix of brightness estimate, of same shape as image
        """

        # Sum over spatial scales
        filters_output = self.bank.apply(image)
        weighted_outputs = self.weight_outputs(filters_output)

        # Sum over scales
        output = weighted_outputs.sum(axis=0)
        return output


class ODOG_RHS2007:
    """Oriented Difference-of-Gaussian model, after Blakeslee & McCourt (1999)

    This model uses 7 Oriented Difference-of-Gaussian filters,
    weighted according to an (approximate) contrast sensitivity function.

    After weighting, each filter output is normalized
    by the global energy of a weighted combination of all other filter outputs.

    This implementation specifically corresponds to Robinson, Hammon and de Sa's (2007)
    implementation, and exactly replicates the available MATLAB version.

    Parameters
    ----------
    shape : array-like[int]
        pixel size (X, Y) of filters
    visextent : array-like[float]
        extent in degrees visual angle, (left, right, top, bottom)

    See also
    --------
    multyscale.filterbanks.RHS2007
    multyscale.normalizaton

    """

    def __init__(self, shape: Sequence[int], visextent: Sequence[float]):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbanks.RHS2007(shape, visextent)

        self.center_sigmas = [sigma[0][0] for sigma in self.bank.sigmas]
        self.weights_slope = 0.1
        self.scale_weights = filterbanks.scale_weights(self.center_sigmas, self.weights_slope)

        self.scale_norm_weights = normalization.scale_norm_weights_equal(len(self.scale_weights))
        self.orientation_norm_weights = normalization.orientation_norm_weights(self.bank.shape[0])
        self.normalization_weights = normalization.create_normalization_weights(
            *self.bank.shape[:2], self.scale_norm_weights, self.orientation_norm_weights
        )

    def weight_outputs(self, filters_output: np.ndarray) -> np.ndarray:
        """Weight filter outputs according to spatial scale of filter

        Uses self.scale_weights for weighting

        Parameters
        ----------
        filters_output : numpy.ndarray
            output of whole filterbank, of shape (O, S, Y, X)

        Returns
        -------
        numpy.ndarray
            weighted output of whole filterbank, same shape is filters_output
        """
        return filterbanks.weight_oriented_multiscale_outputs(filters_output, self.scale_weights)

    def norm_coeffs(self, filters_output: np.ndarray) -> np.ndarray:
        """Construct all normalizing coefficients for given filter outputs

        Uses model.normalization_weights

        Parameters
        ----------
        filters_output : numpy.ndarray
            output of whole filterbank, of shape (O, S, Y, X)

        Returns
        -------
        numpy.ndarray
            all (O', S', Y, X) normalizing coefficients:
            a single 2D (Y, X) weighted combination of all filter outputs
            per (O', S') filter-to-normalize

        See also
        --------
        multyscale.normalization.norm_coeffs
        """
        return normalization.norm_coeffs(filters_output, self.normalization_weights)

    def spatial_kernels(self) -> np.ndarray:
        """Construct all spatial averaging kernels

        ODOG model uses a global image mean kernel

        Returns
        -------
        numpy.ndarray
            all (O, S, Y, X) spatial averaging kernels

        See also
        --------
        multyscale.normalization.spatial_kernel_globalmean
        """
        kernel = normalization.spatial_kernel_globalmean(self.bank.shape[2:])
        kernels = np.ndarray(self.bank.shape[:2] + kernel.shape)
        for o, s in np.ndindex(kernels.shape[:2]):
            kernels[o, s] = kernel

        return kernels

    def norm_energies(self, norm_coeffs: np.ndarray, eps=0.0) -> np.ndarray:
        """Convert normalizing coefficients to energy (denominator for divisive normalization)

        Parameters
        ----------
        norm_coeffs : numpy.ndarray
            all (O, S, Y, X) normalizing coefficients (weighted combination of all filter outputs)
        eps : float, optional
            precision offset, used to avoid square-root of negative numbers, by default 0.0

        Returns
        -------
        numpy.ndarray
            all (O, S, Y, X) normalizing energies (denominator for divisive normalization)

        See also
        --------
        multyscale.models.ODOG_RHS2007.norm_coeffs
        multyscale.normalization.norm_energy
        """
        kernels = self.spatial_kernels()

        norm_energies = np.ndarray(norm_coeffs.shape)
        for o_prime, s_prime in np.ndindex(norm_coeffs.shape[:2]):
            norm_energies[o_prime, s_prime] = normalization.norm_energy(
                norm_coeffs[o_prime, s_prime], kernels[o_prime, s_prime], eps=eps
            )

        return norm_energies

    def normalize_outputs(self, filters_output: np.ndarray, eps=0.0) -> np.ndarray:
        """Apply divisive normalization to given filter outputs

        Parameters
        ----------
        filters_output : numpy.ndarray
            output of whole filterbank, of shape (O, S, Y, X)
        eps : float, optional
            precision offset, used to avoid square-root of negative numbers, by default 0.0

        Returns
        -------
        numpy.ndarray
            all (O, S, Y, X) normalized filter outputs
        """
        norm_coeffs = self.norm_coeffs(filters_output)

        norm_energies = self.norm_energies(norm_coeffs, eps=eps)

        normalized_outputs = normalization.divisive_normalization(
            filters_output, norm_energies, eps=eps
        )

        return normalized_outputs

    def apply(self, image: np.ndarray, eps=0.0) -> np.ndarray:
        """Apply model to given image

        Parameters
        ----------
        image : numpy.ndarray
            image, 2D in shape (Y, X)
        eps : float, optional
            precision offset, used to avoid floating point errors, by default 0.0

        Returns
        -------
        numpy.ndarray
            matrix of brightness estimate, of same shape as image
        """

        # Sum over spatial scales
        filters_output = self.bank.apply(image)
        weighted_outputs = self.weight_outputs(filters_output)

        # Normalize oriented multiscale outputs
        normalized_outputs = self.normalize_outputs(weighted_outputs, eps=eps)

        # Sum over orientations and scales
        output = normalized_outputs.sum((0, 1))

        return output


class LODOG_RHS2007(ODOG_RHS2007):
    """LODOG model, after Robinson, Hammon, and de Sa (2007)

    This model uses 7 Oriented Difference-of-Gaussian filters,
    weighted according to an (approximate) contrast sensitivity function.

    After weighting, each filter output is normalized
    by the local energy of a weighted combination of all other filter outputs.

    This implementation specifically corresponds to Robinson, Hammon and de Sa's (2007)
    implementation, and exactly replicates the available MATLAB version.

    Parameters
    ----------
    shape : array-like[int]
        pixel size (X, Y) of filters
    visextent : array-like[float]
        extent in degrees visual angle, (left, right, top, bottom)
    window_sigma : float
        standard deviation (in degrees) of Gaussian spatial normalization kernel, by default 4.0


    See also
    --------
    multyscale.filterbanks.RHS2007
    multyscale.normalization
    multyscale.normalization.spatial_kernel_gaussian

    """

    def __init__(
        self, shape: Sequence[int], visextent: Sequence[float], window_sigma: float = 4.0
    ):
        super().__init__(shape, visextent)

        self.window_sigma = window_sigma
        self.window_sigmas = np.ones(shape=(*self.bank.shape[:2], 2)) * self.window_sigma

    def spatial_kernels(self) -> np.ndarray:
        """Construct all spatial averaging kernels

        (F)LODOG model uses a 2D Gaussian shape,
        using the model.window_sigma(s)

        Returns
        -------
        numpy.ndarray
            spatial averaging kernel (Y, X)

        See also
        --------
        multyscale.normalization.spatial_kernel_gaussian
        """
        kernels = np.ndarray(self.bank.shape)
        for o_prime, s_prime in np.ndindex(self.window_sigmas.shape[:2]):
            kernels[o_prime, s_prime] = normalization.spatial_kernel_gaussian(
                self.bank.x,
                self.bank.y,
                sigmas=self.window_sigmas[o_prime, s_prime],
            )
        return kernels


class FLODOG_RHS2007(LODOG_RHS2007):
    """LODOG model, after Robinson, Hammon, and de Sa (2007)

    This model uses 7 Oriented Difference-of-Gaussian filters,
    weighted according to an (approximate) contrast sensitivity function.

    After weighting, each filter output is normalized
    by the local energy of a weighted combination of all other filter outputs.

    This implementation specifically corresponds to Robinson, Hammon and de Sa's (2007)
    implementation, and exactly replicates the available MATLAB version.

    Parameters
    ----------
    shape : array-like[int]
        pixel size (X, Y) of filters
    visextent : array-like[float]
        extent in degrees visual angle, (left, right, top, bottom)
    sdmix : float
        standard deviation of Gaussian weighting function, by default 0.5
    spatial_window_scalar : float
        scaling factor of Gaussian spatial normalization kernel relative to filter size,
        by default 4.0


    See also
    --------
    multyscale.filterbanks.RHS2007
    multyscale.normalization
    multyscale.normalization.spatial_kernel_gaussian

    """

    def __init__(
        self,
        shape: Sequence[int],
        visextent: Sequence[float],
        sdmix: float = 0.5,
        spatial_window_scalar: float = 4.0,
    ):
        super().__init__(shape, visextent)

        self.sdmix = sdmix  # stdev of Gaussian weights for scale mixing
        self.scale_norm_weights = normalization.scale_norm_weights_gaussian(
            len(self.scale_weights), self.sdmix
        )
        self.normalization_weights = normalization.create_normalization_weights(
            *self.bank.shape[:2], self.scale_norm_weights, self.orientation_norm_weights
        )

        self.spatial_window_scalar = spatial_window_scalar
        self.window_sigmas = self.spatial_window_scalar * np.broadcast_to(
            np.array(self.center_sigmas)[None, ..., None], (*self.bank.shape[:2], 2)
        )
