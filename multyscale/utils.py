# Third party imports
import numpy as np


def octave_intervals(num):
    # TODO: docstring

    # Octave intervals means x_i : x_i+1 = 1:2
    # So, log spacing in base 2
    # num values, from 1
    x = np.logspace(1, num, num=num, base=2)

    # Normalize, so that maximum is 1
    return x/x.max()
