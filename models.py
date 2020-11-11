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
