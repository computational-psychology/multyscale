import numpy as np
import filters


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
