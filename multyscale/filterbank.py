# Third party imports
import numpy as np

# Local application imports
from . import filters, utils

# TODO: (abstract) base class Filterbank, with apply method

# TODO: data structure for filter output, with filter metadata


# Weighting filter(outputs) according to spatial scale
# Defining the scale weights
def scale_weights(center_sigmas, slope):
    # TODO: docstring
    scale_weights = (1 / np.asarray(center_sigmas)) ** slope
    return scale_weights


# Weighting multiscale output
def weight_multiscale_outputs(multiscale_filters, scale_weights):
    # TODO: docstring
    # Weight multiscale filters outputs according to filter spatial scale
    weighted_multiscale = [f * w for f, w in zip(multiscale_filters, scale_weights)]
    return np.asarray(weighted_multiscale)


# Weighting oriented multiscales
def weight_oriented_multiscale_outputs(filters_output, scale_weights):
    # TODO: docstring
    # Weight each filter output according to filter spatial scale
    weighted_filters_output = [
        weight_multiscale_outputs(m, scale_weights) for m in filters_output
    ]
    return np.asarray(weighted_filters_output)


class DOGBank:
    def __init__(self, sigmas, x, y):
        # TODO: docstring
        # TODO: typehints
        self.sigmas = sigmas
        self.x = x
        self.y = y

        self.filters = np.empty((len(sigmas), x.shape[0], x.shape[1]))
        for i, sigma in enumerate(sigmas):
            dog = filters.dog(x, y, sigma)
            self.filters[i, :, :] = dog

    def apply(self, image):
        # TODO: docstring
        # TODO: typehints
        filters_output = np.empty(self.filters.shape)
        for i in range(self.filters.shape[0]):
            filters_output[i, ...] = filters.apply(
                image, self.filters[i, ...], pad=True
            )
        return filters_output


class ODOGBank:
    def __init__(self, orientations, sigmas, x, y):
        # TODO: docstring
        # TODO: typehints
        # TODO: don't store filtes as ND-array
        self.orientations = orientations
        self.sigmas = sigmas
        self.x = x
        self.y = y

        self.filters = np.empty(
            (len(orientations), len(sigmas), x.shape[0], x.shape[1])
        )

        for i, angle in enumerate(orientations):
            for j, sigma in enumerate(sigmas):
                odog = filters.odog(x, y, sigma, (angle, angle))
                self.filters[i, j, :, :] = odog

    def apply(self, image):
        # TODO: docstring
        # TODO: typehints
        filters_output = np.empty(self.filters.shape)
        for i in range(self.filters.shape[0]):
            for j in range(self.filters.shape[1]):
                filters_output[i, j, ...] = filters.apply(
                    image, self.filters[i, j, ...], pad=True
                )
        return filters_output


def BM1999(filtershape=(1024, 1024), visextent=(24, 24)):
    # TODO: docstring
    # TODO: typehints
    # TODO: figure out actual BM space parameter....

    # Parameters (BM1999)
    n_orientations = 6
    num_scales = 7
    largest_center_sigma = 3  # in degrees
    center_sigmas = utils.octave_intervals(num_scales) * largest_center_sigma
    cs_ratio = 2  # center-surround ratio

    # Convert to filterbank parameters
    orientations = np.arange(0, 180, 180 / n_orientations)
    sigmas = [((s, s), (s, cs_ratio * s)) for s in center_sigmas]

    # Create image coordinate system:
    axish = np.linspace(visextent[0], visextent[1], filtershape[0])
    axisv = np.linspace(visextent[2], visextent[3], filtershape[1])
    (x, y) = np.meshgrid(axish, axisv)

    # Create filterbank
    bank = ODOGBank(orientations, sigmas, x, y)
    return bank
