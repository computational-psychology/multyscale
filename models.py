import numpy as np
import filterbank

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
