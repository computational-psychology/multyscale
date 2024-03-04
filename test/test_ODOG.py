# %% Imports
import numpy as np
import pytest
import RHS_implementation

import multyscale

# %% Parameters of image
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
# NOTE: RHS implementation doesn't actually use (-16,16,-16,16)


# %% Model
@pytest.fixture
def model(stimulus):
    return multyscale.models.ODOG_RHS2007(stimulus.shape, visextent)


# %% Tests
def test_filters(MATLAB_filteroutput, model, stimulus):
    filter_output = model.bank.apply(stimulus)

    assert np.allclose(MATLAB_filteroutput, filter_output)


def test_weights(model):
    assert np.allclose(model.scale_weights, RHS_implementation.w_val)


def test_normalize(output_ODOG_MATLAB, model, MATLAB_filteroutput):
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)
    normed_outputs = model.normalize_outputs(weighted_outputs)

    output = np.sum(normed_outputs, (0, 1))

    assert np.allclose(output, output_ODOG_MATLAB)


def test_model_output(output_ODOG_MATLAB, model, stimulus):
    output = model.apply(stimulus)
    assert np.allclose(output, output_ODOG_MATLAB)
