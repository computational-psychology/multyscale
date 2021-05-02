# Third party imports
import numpy as np

# Local application imports
from . import filterbank, filters, normalization

# TODO: refactor filter-output datastructures


class ODOG_BM1999:
    # TODO: docstring

    def __init__(self, shape, visextent):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbank.BM1999(shape, visextent)

        self.center_sigmas = [sigma[0][0] for sigma in self.bank.sigmas]
        self.weights_slope = 0.1
        self.scale_weights = filterbank.scale_weights(
            self.center_sigmas, self.weights_slope
        )

        self.scale_norm_weights = normalization.scale_norm_weights_equal(
            len(self.scale_weights)
        )
        self.orientation_norm_weights = normalization.orientation_norm_weights(6)
        self.normalization_weights = normalization.create_normalization_weights(
            6, 7, self.scale_norm_weights, self.orientation_norm_weights
        )

    def weight_outputs(self, filters_output):
        return filterbank.weight_oriented_multiscale_outputs(
            filters_output, self.scale_weights
        )

    def normalizers(self, filters_output):
        # Get normalizers
        normalizers = normalization.normalizers(
            filters_output, self.normalization_weights
        )
        return normalizers

    def normalizers_to_RMS(self, normalizers):
        # Get RMS from each normalizer
        spatial_avg_filters = normalization.spatial_avg_windows_globalmean(normalizers)
        normalizer_RMS = normalization.nomalizers_to_RMS(
            normalizers, spatial_avg_filters
        )
        return normalizer_RMS

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


class LODOG_RHS2007(ODOG_BM1999):
    # TODO: docstring

    def __init__(self, shape, visextent):
        self.window_sigma = 2
        super().__init__(shape, visextent)

    def normalizers_to_RMS(self, normalizers):
        # Create Gaussian window
        window = filters.gaussian2d(
            self.bank.x, self.bank.y, (self.window_sigma, self.window_sigma)
        )

        # Normalize window to unit-sum (== spatial averaging filter)
        window = window / window.sum()

        # Get RMS from each normalizer
        normalizer_RMS = np.ndarray(normalizers.shape)
        for o, s in np.ndindex(normalizers.shape[:2]):
            normalizer = normalizers[o, s]

            # Square image
            normalizer = np.square(normalizer)

            # Apply Gaussian window
            normalizer = filters.apply(normalizer, window)

            # Square root
            normalizer = np.sqrt(normalizer)

            normalizer_RMS[o, s, ...] = normalizer
        return normalizer_RMS


class FLODOG_RHS2007(LODOG_RHS2007):
    # TODO: docstring

    def __init__(self, shape, visextent):
        super().__init__(shape, visextent)

        self.sdmix = 0.5  # stdev of Gaussian weights for scale mixing
        self.scale_norm_weights = normalization.scale_norm_weights_gaussian(
            len(self.scale_weights), self.sdmix
        )
        self.normalization_weights = normalization.create_normalization_weights(
            6, 7, self.scale_norm_weights, self.orientation_norm_weights
        )
