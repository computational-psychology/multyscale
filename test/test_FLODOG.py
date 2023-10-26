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
def model(stimulus, MATLAB_FLODOG_params):
    return multyscale.models.FLODOG_RHS2007(
        stimulus.shape,
        visextent,
        spatial_window_scalar=MATLAB_FLODOG_params["sigx"],
        sdmix=MATLAB_FLODOG_params["sdmix"],
    )


# %% Tests
def test_filters(MATLAB_filteroutput, model, stimulus):
    filter_output = model.bank.apply(stimulus)

    assert np.allclose(MATLAB_filteroutput, filter_output)


def test_weights(model):
    assert np.allclose(model.scale_weights, RHS_implementation.w_val)


def test_scale_norm_weights(model, MATLAB_FLODOG_params):
    RHS_weights = RHS_implementation.FLODOG_normweights(sdmix=MATLAB_FLODOG_params["sdmix"])
    assert np.allclose(RHS_weights.shape, model.scale_norm_weights.shape)
    assert np.allclose(RHS_weights, model.scale_norm_weights)


def test_normalizers(model, MATLAB_filteroutput, MATLAB_FLODOG_params):
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)
    normalizers = model.normalizers(weighted_outputs)
    RHS_norms = RHS_implementation.FLODOG_normalizers(
        weighted_outputs, sdmix=MATLAB_FLODOG_params["sdmix"]
    )
    assert np.allclose(normalizers.shape, RHS_norms.shape)
    assert np.allclose(normalizers, RHS_norms)


def test_spatial_masks(model, MATLAB_FLODOG_params):
    # Is the spatial (Gaussian) averaging window the same?
    masks = RHS_implementation.FLODOG_masks(sigx=MATLAB_FLODOG_params["sigx"])
    spatial_avg_filters = multyscale.normalization.spatial_avg_windows_gaussian(
        model.bank.x, model.bank.y, model.window_sigmas
    )
    assert np.allclose(spatial_avg_filters, masks)


def test_FLODOG_RMS(model, MATLAB_filteroutput, MATLAB_FLODOG_params):
    weighted_outputs = RHS_implementation.weight(MATLAB_filteroutput)
    normalizers = RHS_implementation.FLODOG_normalizers(
        weighted_outputs, sdmix=MATLAB_FLODOG_params["sdmix"]
    )
    RHS_RMSs = RHS_implementation.FLODOG_RMSs(normalizers, sigx=MATLAB_FLODOG_params["sigx"])

    RMSs = model.normalizers_to_RMS(normalizers)
    assert np.allclose(RMSs, RHS_RMSs)


def test_normalized_outputs(model, MATLAB_filteroutput, MATLAB_FLODOG_params):
    weighted_outputs = RHS_implementation.weight(MATLAB_filteroutput)
    RHS_normalized_outputs = RHS_implementation.flodog_normalize(
        weighted_outputs,
        sigx=MATLAB_FLODOG_params["sigx"],
        sdmix=MATLAB_FLODOG_params["sdmix"],
    )

    normed_outputs = model.normalize_outputs(weighted_outputs)

    assert np.allclose(normed_outputs, RHS_normalized_outputs)


def test_model_output(output_FLODOG_MATLAB, model, stimulus):
    output = model.apply(stimulus)
    assert np.allclose(output, output_FLODOG_MATLAB)
