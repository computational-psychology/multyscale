# Third party imports
import numpy as np

# Local application imports
from . import filterbank, filters, normalization

# TODO: refactor filter-output datastructures


class DOG_BM1997:
    def __init__(self, shape, visextent):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbank.BM1997(shape, visextent)

        self.center_sigmas = [sigma[0] for sigma in self.bank.sigmas]
        self.weights_slope = 0.1
        self.scale_weights = filterbank.scale_weights(self.center_sigmas, self.weights_slope)

    def weight_outputs(self, filters_output):
        return filterbank.weight_multiscale_outputs(filters_output, self.scale_weights)

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

        self.bank = filterbank.RHS2007(shape, visextent)

        self.center_sigmas = [sigma[0][0] for sigma in self.bank.sigmas]
        self.weights_slope = 0.1
        self.scale_weights = filterbank.scale_weights(self.center_sigmas, self.weights_slope)

        self.scale_norm_weights = normalization.scale_norm_weights_equal(len(self.scale_weights))
        self.orientation_norm_weights = normalization.orientation_norm_weights(self.bank.shape[0])
        self.normalization_weights = normalization.create_normalization_weights(
            *self.bank.shape[:2], self.scale_norm_weights, self.orientation_norm_weights
        )

    def weight_outputs(self, filters_output):
        return filterbank.weight_oriented_multiscale_outputs(filters_output, self.scale_weights)

    def normalizers(self, filters_output):
        # Get normalizers
        normalizers = normalization.normalizers(filters_output, self.normalization_weights)
        return normalizers

    def normalizers_to_RMS(self, normalizers):
        # Get RMS from each normalizer
        spatial_avg_filters = normalization.spatial_avg_windows_globalmean(normalizers)
        normalizers_RMS = normalizers.copy()
        normalizers_RMS = np.square(normalizers_RMS)
        normalizers_RMS = spatial_avg_filters * normalizers_RMS
        normalizers_RMS = normalizers_RMS.sum(axis=(-1, -2))
        normalizers_RMS = np.sqrt(normalizers_RMS)
        return normalizers_RMS

    def normalize_outputs(self, filters_output):
        # TODO: docstring
        normalizers = self.normalizers(filters_output)

        normalizer_RMS = self.normalizers_to_RMS(normalizers)

        normalized_outputs = np.ndarray(filters_output.shape)
        for o, s in np.ndindex(filters_output.shape[:2]):
            normalized_outputs[o, s] = filters_output[o, s] / normalizer_RMS[o, s]

        return normalized_outputs

    def apply(self, image):
        # TODO: docstring

        # Sum over spatial scales
        filters_output = self.bank.apply(image)
        weighted_outputs = self.weight_outputs(filters_output)

        # Normalize oriented multiscale outputs
        normalized_outputs = self.normalize_outputs(weighted_outputs)

        # Sum over orientations and scales
        output = normalized_outputs.sum((0, 1))
        return output


class LODOG_RHS2007(ODOG_RHS2007):
    # TODO: docstring

    def __init__(self, shape, visextent, window_sigma=4):
        super().__init__(shape, visextent)

        self.window_sigma = window_sigma
        self.window_sigmas = np.ones(shape=(*self.bank.shape[:2], 2)) * self.window_sigma

    def normalizers_to_RMS(self, normalizers):
        # Expand sigmas
        # Get RMS from each normalizer
        spatial_avg_filters = normalization.spatial_avg_windows_gaussian(
            self.bank.x, self.bank.y, self.window_sigmas
        )
        normalizers_RMS = normalizers.copy()
        normalizers_RMS = np.square(normalizers_RMS)
        for o, s in np.ndindex(normalizers_RMS.shape[:2]):
            normalizers_RMS[o, s] = filters.apply(
                spatial_avg_filters[o, s], normalizers_RMS[o, s], padval=0
            )
        normalizers_RMS += 1e-6
        normalizers_RMS = np.sqrt(normalizers_RMS)
        normalizers_RMS += 1e-6
        return normalizers_RMS


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
