# %% Imports
import numpy as np

from multyscale import models

# %% Parameters of image
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
# NOTE: RHS implementation doesn't actually use (-16,16,-16,16)


# %% Tests
def test_odog_output(output_ODOG_MATLAB, stimulus):
    model = models.ODOG_RHS2007(stimulus.shape, visextent)
    output = model.apply(stimulus)
    assert np.allclose(output, output_ODOG_MATLAB)


def test_lodog_output(stimulus, output_LODOG_MATLAB, MATLAB_LODOG_params):
    model = models.LODOG_RHS2007(
        stimulus.shape,
        visextent,
        window_sigma=MATLAB_LODOG_params["sig1"] / 32,
    )
    output = model.apply(stimulus, eps=1e-6)
    assert np.allclose(output, output_LODOG_MATLAB)


def test_flodog_output(stimulus, output_FLODOG_MATLAB, MATLAB_FLODOG_params):
    model = models.FLODOG_RHS2007(
        stimulus.shape,
        visextent,
        sdmix=MATLAB_FLODOG_params["sdmix"],
        spatial_window_scalar=MATLAB_FLODOG_params["sigx"],
    )
    output = model.apply(stimulus, eps=1e-6)
    assert np.allclose(output, output_FLODOG_MATLAB)
