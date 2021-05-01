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

    def normalize_outputs(self, filters_output):
        # TODO: docstring
        # Get normalizers
        normalizers = normalization.normalizers(
            filters_output, self.normalization_weights
        )

        # Get RMS from each normalizer
        normalizer_RMS = np.sqrt(np.square(normalizers).mean((-1, -2)))

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

    def normalize_outputs(self, filters_output):
        # TODO: docstring
        # Get normalizers
        normalizers = normalization.normalizers(
            filters_output, self.normalization_weights
        )

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

        # Normalize
        normalized_outputs = np.ndarray(filters_output.shape)
        for o, s in np.ndindex(filters_output.shape[:2]):
            normalized_outputs[o, s] = filters_output[o, s] / normalizer_RMS[o, s]

        return normalized_outputs


class FLODOG_RHS2007:
    # TODO: docstring

    def __init__(self, shape, visextent):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbank.BM1999(shape, visextent)

        center_sigmas = [sigma[0][0] for sigma in self.bank.sigmas]
        self.weights_slope = 0.1
        self.scale_weights = (1 / np.asarray(center_sigmas)) ** self.weights_slope

        self.window_sigma = 2
        self.sdmix = 0.5  # stdev of Gaussian weights for scale mixing

    def weight_filters_output(self, filters_output):
        # TODO: docstring
        # Weight each filter output according to scale
        weighted_filters_output = np.empty(filters_output.shape)
        for i in range(filters_output.shape[0]):
            for j, output in enumerate(filters_output[i, ...]):
                weighted_filters_output[i, j, ...] = output * self.scale_weights[j]
        return weighted_filters_output

    def create_normalizers(self, filters_output):
        # TODO: docstring
        # Create normalizer images
        normalizers = np.empty(filters_output.shape)
        for o, multiscale in enumerate(filters_output):
            for i, filt in enumerate(multiscale):
                normalizer = np.empty(filt.shape)

                # Identify relative index of each scale to the current one
                rel_i = i - np.asarray(range(multiscale.shape[0]))

                # Gaussian weights, based on relative index
                gweights = np.exp(-(rel_i ** 2) / 2 * self.sdmix ** 2) / (
                    self.sdmix * np.sqrt(2 * np.pi)
                )

                # Sum filter outputs, by Gaussian weights
                normalizer = np.tensordot(multiscale, gweights, axes=(0, 0))

                # Normalize normalizer...
                area = gweights.sum()
                normalizer = normalizer / area

                # Accumulate
                normalizers[o, i, ...] = normalizer
        return normalizers

    def blur_normalizers(self, normalizers):
        # TODO: docstring
        # Create Gaussian window
        window = filters.gaussian2d(
            self.bank.x, self.bank.y, (self.window_sigma, self.window_sigma)
        )

        # Normalize window to unit-sum (== spatial averaging filter)
        window = window / window.sum()

        for o, multiscale in enumerate(normalizers):
            for s, normalizer in enumerate(multiscale):
                # Square image
                normalizer = np.square(normalizer)

                # Apply Gaussian window
                normalizer = filters.apply(normalizer, window)

                # Square root
                normalizer = np.sqrt(normalizer)
                normalizers[o, s, ...] = normalizer
        return normalizers

    def normalize_filters_output(self, filters_output, normalizers):
        # TODO: docstring
        normalized_outputs = np.empty(filters_output.shape)
        for o, multiscale in enumerate(filters_output):
            for s, output in enumerate(multiscale):
                normalized_outputs[o, s] = output / normalizers[o, s]
        return normalized_outputs

    def apply(self, image):
        # TODO: docstring

        # Apply filterbank
        filters_output = self.bank.apply(image)

        # Weight filter output
        filters_output = self.weight_filters_output(filters_output)

        # Normalize filtes output
        normalizers = self.create_normalizers(filters_output)
        normalizers = self.blur_normalizers(normalizers)
        normalized_outputs = self.normalize_filters_output(filters_output, normalizers)

        # Sum outputs
        output = normalized_outputs.sum((0, 1))
        return output
