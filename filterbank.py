import numpy as np
import filters
import utils

# TODO: (abstract) base class Filterbank, with apply method

# TODO: data structure for filter output, with filter metadata


def apply(image, bank):
    # TODO: docstring
    # TODO: typehints
    # TODO: refactor as method of filterbank class
    filters_output = np.empty(bank.shape)
    for i in range(bank.shape[0]):
        for j in range(bank.shape[1]):
            filters_output[i, j, ...] = filters.apply(image, bank[i, j, ...])
    return filters_output


def odog_bank(orientations,
              sigmas,
              x, y):
    # TODO: docstring
    # TODO: typehints
    # TODO: refactor as class
    # TODO: don't store filtes as ND-array
    bank = np.empty((len(orientations),
                    len(sigmas),
                    x.shape[0], x.shape[1]))

    for i, angle in enumerate(orientations):
        for j, sigma in enumerate(sigmas):
            odog = filters.odog(x, y,
                                sigma,
                                (angle, angle))
            bank[i, j, :, :] = odog

    return bank


def BM1999(filtershape=(1024, 1024),
           visextent=(24, 24)):
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
    orientations = np.arange(0, 180, 180/n_orientations)
    sigmas = [((s, s), (s, cs_ratio*s)) for s in center_sigmas]

    # Create image coordinate system:
    axish = np.linspace(visextent[0], visextent[1], filtershape[0])
    axisv = np.linspace(visextent[2], visextent[3], filtershape[1])
    (x, y) = np.meshgrid(axish, axisv)

    # Create filterbank
    bank = odog_bank(orientations, sigmas, x, y)
    return bank
