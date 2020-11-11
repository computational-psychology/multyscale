import numpy as np
import filters

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
