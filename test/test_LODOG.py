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
def model(stimulus, MATLAB_LODOG_params):
    return multyscale.models.LODOG_RHS2007(
        stimulus.shape,
        visextent,
        window_sigma=MATLAB_LODOG_params["sig1"] / 32,
    )


# %% Tests
def test_filters(MATLAB_filteroutput, model, stimulus):
    filter_output = model.bank.apply(stimulus)

    assert np.allclose(MATLAB_filteroutput, filter_output)


def test_weights(model):
    assert np.allclose(model.scale_weights, RHS_implementation.w_val)


def test_normalizers(model, MATLAB_filteroutput):
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)
    normalizers = model.normalizers(weighted_outputs)
    RHS_norms = RHS_implementation.LODOG_normalizers(weighted_outputs)
    assert np.allclose(normalizers.shape, RHS_norms.shape)
    assert np.allclose(normalizers, RHS_norms)


def test_spatial_mask(model, MATLAB_LODOG_params):
    # Is the spatial (Gaussian) averaging window the same?
    RHS_mask = RHS_implementation.LODOG_mask(sig1=MATLAB_LODOG_params["sig1"])
    spatial_avg_filters = multyscale.normalization.spatial_avg_windows_gaussian(
        model.bank.x, model.bank.y, model.window_sigmas
    )
    for o, s in np.ndindex(spatial_avg_filters.shape[:2]):
        assert np.allclose(spatial_avg_filters[o, s], RHS_mask)


def test_LODOG_RMS(model, MATLAB_filteroutput, MATLAB_LODOG_params):
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)
    normalizers = model.normalizers(weighted_outputs)

    RMSs = model.normalizers_to_RMS(normalizers)
    RHS_RMSs = RHS_implementation.LODOG_RMSs(normalizers, sig1=MATLAB_LODOG_params["sig1"])
    assert np.allclose(RMSs, RHS_RMSs)


def test_normalized_outputs(model, MATLAB_filteroutput, MATLAB_LODOG_params):
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)

    normed_outputs = model.normalize_outputs(weighted_outputs)

    RHS_normalized_outputs = RHS_implementation.LODOG_normalize(
        weighted_outputs, sig1=MATLAB_LODOG_params["sig1"]
    )
    assert np.allclose(normed_outputs, RHS_normalized_outputs)


def test_model_output(output_LODOG_MATLAB, model, stimulus):
    output = model.apply(stimulus)
    assert np.allclose(output, output_LODOG_MATLAB)
