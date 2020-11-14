import numpy as np
import filterbank
import filters

# TODO: refactor filter-output datastructures


class ODOG_BM1999:
    # TODO: docstring

    def __init__(self,
                 shape, visextent):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbank.BM1999(shape, visextent)

        center_sigmas = [sigma[0][0] for sigma in self.bank.sigmas]
        self.weights_slope = .1
        self.scale_weights = (1 / np.asarray(center_sigmas)) \
            ** self.weights_slope

    def sum_scales(self, filters_output):
        # TODO: docstring
        multiscale_output = np.tensordot(filters_output,
                                         self.scale_weights,
                                         (1, 0))
        return multiscale_output

    def normalize_multiscale_output(self, multiscale_output):
        # TODO: docstring
        normalized_multiscale_output = np.empty(multiscale_output.shape)
        for i in range(multiscale_output.shape[0]):
            image = multiscale_output[i]
            rms = np.sqrt(np.square(image).mean((-1, -2)))  # image-wide RMS
            normalized_multiscale_output[i] = image / rms
        return normalized_multiscale_output

    def apply(self, image):
        # TODO: docstring

        # Sum over spatial scales
        filters_output = self.bank.apply(image)
        multiscale_output = self.sum_scales(filters_output)

        # Normalize oriented multiscale outputs
        normalized_multiscale_output = \
            self.normalize_multiscale_output(multiscale_output)

        # Sum over orientations
        output = normalized_multiscale_output.sum(0)
        return output


class LODOG_RHS2007:
    # TODO: docstring

    def __init__(self,
                 shape, visextent):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbank.BM1999(shape, visextent)

        center_sigmas = [sigma[0][0] for sigma in self.bank.sigmas]
        self.weights_slope = .1
        self.scale_weights = (1 / np.asarray(center_sigmas)) \
            ** self.weights_slope

        self.window_sigma = 2

    def sum_scales(self, filters_output):
        # TODO: docstring
        multiscale_output = np.tensordot(filters_output,
                                         self.scale_weights,
                                         (1, 0))
        return multiscale_output

    def normalize_multiscale_output(self, multiscale_output):
        # TODO: docstring
        # Create Gaussian window
        window = filters.gaussian2d(self.bank.x, self.bank.y,
                                    (self.window_sigma, self.window_sigma))

        # Normalize window to unit-sum (== spatial averaging filter)
        window = window / window.sum()

        # Create normalizer images
        normalized_multiscale_output = np.empty(multiscale_output.shape)
        normalizers = np.empty(multiscale_output.shape)
        for i, image in enumerate(multiscale_output):
            # Square image
            normalizer = np.square(image)

            # Apply Gaussian window
            normalizer = filters.apply(normalizer, window)

            # Square root
            normalizer = np.sqrt(normalizer)
            normalizers[i, ...] = normalizer

            # Normalize
            normalized_multiscale_output[i, ...] = image / normalizer
        return normalized_multiscale_output, normalizers

    def apply(self, image):
        # TODO: docstring

        # Sum over spatial scales
        filters_output = self.bank.apply(image)
        multiscale_output = self.sum_scales(filters_output)

        # Normalize oriented multiscale outputs
        normalized_multiscale_output = \
            self.normalize_multiscale_output(multiscale_output)[0]

        # Sum over orientations
        output = normalized_multiscale_output.sum(0)
        return output


class FLODOG_RHS2007:
    # TODO: docstring

    def __init__(self,
                 shape, visextent):
        self.shape = shape
        self.visextent = visextent

        self.bank = filterbank.BM1999(shape, visextent)

        center_sigmas = [sigma[0][0] for sigma in self.bank.sigmas]
        self.weights_slope = .1
        self.scale_weights = (1 / np.asarray(center_sigmas)) \
            ** self.weights_slope

        self.window_sigma = 2
        self.sdmix = .5  # stdev of Gaussian weights for scale mixing

    def weight_filters_output(self, filters_output):
        # TODO: docstring
        # Weight each filter output according to scale
        weighted_filters_output = np.empty(filters_output.shape)
        for i in range(filters_output.shape[0]):
            for j, output in enumerate(filters_output[i, ...]):
                weighted_filters_output[i, j, ...] = output * \
                    self.scale_weights[j]
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
                gweights = np.exp(- rel_i ** 2 / 2 * self.sdmix ** 2) /\
                    (self.sdmix * np.sqrt(2 * np.pi))

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
        window = filters.gaussian2d(self.bank.x, self.bank.y,
                                    (self.window_sigma, self.window_sigma))

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
        output = normalized_outputs.sum((0,1))
        return output
