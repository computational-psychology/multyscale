# %% Imports
import numpy as np

from multyscale import models

# %% Parameters of image
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
# NOTE: RHS implementation doesn't actually use (-16,16,-16,16)


# %% Tests
def test_ODOG(output_ODOG_MATLAB, stimulus):
    model = models.ODOG_RHS2007(stimulus.shape, visextent)
    output = model.apply(stimulus)
    assert np.allclose(output, output_ODOG_MATLAB)


def test_LODOG(stimulus, output_LODOG_MATLAB, params_LODOG_MATLAB):
    model = models.LODOG_RHS2007(
        stimulus.shape,
        visextent,
        window_sigma=params_LODOG_MATLAB["sig1"] / 32,
    )
    output = model.apply(stimulus, eps=1e-6)
    assert np.allclose(output, output_LODOG_MATLAB)


def test_FLODOG(stimulus, output_FLODOG_MATLAB, params_FLODOG_MATLAB):
    model = models.FLODOG_RHS2007(
        stimulus.shape,
        visextent,
        sdmix=params_FLODOG_MATLAB["sdmix"],
        spatial_window_scalar=params_FLODOG_MATLAB["sigx"],
    )
    output = model.apply(stimulus, eps=1e-6)
    assert np.allclose(output, output_FLODOG_MATLAB)
