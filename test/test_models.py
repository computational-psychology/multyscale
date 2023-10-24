# %% Imports
import numpy as np

from multyscale import models

# %% Parameters of image
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
# NOTE: RHS implementation doesn't actually use (-16,16,-16,16)


# %% Tests
def test_odog_output(output_odog_MATLAB, stimulus):
    model = models.ODOG_RHS2007(stimulus.shape, visextent)
    output = model.apply(stimulus)
    assert np.allclose(output, output_odog_MATLAB)


def test_lodog_output(stimulus, output_lodog_MATLAB):
    window_sigma = 128 / 32

    model = models.LODOG_RHS2007(stimulus.shape, visextent, window_sigma=window_sigma)
    output = model.apply(stimulus)
    assert np.allclose(output, output_lodog_MATLAB)


def test_flodog_output(stimulus, output_flodog_MATLAB):
    windowSizeScalar = 4
    m = 0.5

    model = models.FLODOG_RHS2007(stimulus.shape, visextent)
    output = model.apply(stimulus)
    assert np.allclose(output, output_flodog_MATLAB)
