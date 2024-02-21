# Third party imports
import numpy as np

# Local application imports
from . import filterbanks, normalization

# TODO: refactor filter-output datastructures


class DOG_BM1997:
    def __init__(self, shape, visextent):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbanks.BM1997(shape, visextent)

        self.center_sigmas = [sigma[0] for sigma in self.bank.sigmas]
        self.weights_slope = 0.1
        self.scale_weights = filterbanks.scale_weights(self.center_sigmas, self.weights_slope)

    def weight_outputs(self, filters_output):
        return filterbanks.weight_multiscale_outputs(filters_output, self.scale_weights)

    def apply(self, image):
        # TODO: docstring

        # Sum over spatial scales
        filters_output = self.bank.apply(image)
        weighted_outputs = self.weight_outputs(filters_output)

        # Sum over scales
        output = weighted_outputs.sum(axis=0)
        return output


class ODOG_RHS2007:
    # TODO: docstring

    def __init__(self, shape, visextent):
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

    def weight_outputs(self, filters_output):
        return filterbanks.weight_oriented_multiscale_outputs(filters_output, self.scale_weights)

    def norm_coeffs(self, filters_output):
        return normalization.norm_coeffs(filters_output, self.normalization_weights)

    def spatial_kernels(self):
        kernel = normalization.spatial_kernel_globalmean(self.bank.shape[2:])
        kernels = np.ndarray(self.bank.shape[:2] + kernel.shape)
        for o, s in np.ndindex(kernels.shape[:2]):
            kernels[o, s] = kernel

        return kernels

    def norm_energies(self, norm_coeffs, eps=0.0):
        kernels = self.spatial_kernels()

        norm_energies = np.ndarray(norm_coeffs.shape)
        for o_prime, s_prime in np.ndindex(norm_coeffs.shape[:2]):
            norm_energies[o_prime, s_prime] = normalization.norm_energy(
                norm_coeffs[o_prime, s_prime], kernels[o_prime, s_prime], eps=eps
            )

        return norm_energies

    def normalize_outputs(self, filters_output, eps=0.0):
        # TODO: docstring
        norm_coeffs = self.norm_coeffs(filters_output)

        norm_energies = self.norm_energies(norm_coeffs, eps=eps)

        normalized_outputs = normalization.divisive_normalization(
            filters_output, norm_energies, eps=eps
        )

        return normalized_outputs

    def apply(self, image, eps=0.0):
        # TODO: docstring

        # Sum over spatial scales
        filters_output = self.bank.apply(image)
        weighted_outputs = self.weight_outputs(filters_output)

        # Normalize oriented multiscale outputs
        normalized_outputs = self.normalize_outputs(weighted_outputs, eps=eps)

        # Sum over orientations and scales
        output = normalized_outputs.sum((0, 1))

        return output


class LODOG_RHS2007(ODOG_RHS2007):
    # TODO: docstring

    def __init__(self, shape, visextent, window_sigma=4):
        super().__init__(shape, visextent)

        self.window_sigma = window_sigma
        self.window_sigmas = np.ones(shape=(*self.bank.shape[:2], 2)) * self.window_sigma

    def spatial_kernels(self):
        kernels = np.ndarray(self.bank.shape)
        for o_prime, s_prime in np.ndindex(self.window_sigmas.shape[:2]):
            kernels[o_prime, s_prime] = normalization.spatial_kernel_gaussian(
                self.bank.x,
                self.bank.y,
                sigmas=self.window_sigmas[o_prime, s_prime],
            )
        return kernels


class FLODOG_RHS2007(LODOG_RHS2007):
    # TODO: docstring

    def __init__(self, shape, visextent, sdmix=0.5, spatial_window_scalar=4):
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
