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
def model(stimulus, params_FLODOG_MATLAB):
    return multyscale.models.FLODOG_RHS2007(
        stimulus.shape,
        visextent,
        spatial_window_scalar=params_FLODOG_MATLAB["sigx"],
        sdmix=params_FLODOG_MATLAB["sdmix"],
    )


# %% Tests
def test_filters(MATLAB_filteroutput, model, stimulus):
    filter_output = model.bank.apply(stimulus)

    assert np.allclose(MATLAB_filteroutput, filter_output)


def test_weights(model):
    assert np.allclose(model.scale_weights, RHS_implementation.w_val)


def test_scale_norm_weights(model, params_FLODOG_MATLAB):
    RHS_weights = RHS_implementation.FLODOG_normweights(sdmix=params_FLODOG_MATLAB["sdmix"])
    assert np.allclose(RHS_weights.shape, model.scale_norm_weights.shape)
    assert np.allclose(RHS_weights, model.scale_norm_weights)


def test_norm_coeffs(model, MATLAB_filteroutput, params_FLODOG_MATLAB):
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)
    norm_coeffs = model.norm_coeffs(weighted_outputs)
    RHS_norms = RHS_implementation.FLODOG_normalizers(
        weighted_outputs, sdmix=params_FLODOG_MATLAB["sdmix"]
    )
    assert np.allclose(norm_coeffs.shape, RHS_norms.shape)
    assert np.allclose(norm_coeffs, RHS_norms)


def test_spatial_masks(model, params_FLODOG_MATLAB):
    # Is the spatial (Gaussian) averaging window the same?
    RHS_kernels = RHS_implementation.FLODOG_masks(sigx=params_FLODOG_MATLAB["sigx"])
    spatial_kernels = model.spatial_kernels()
    for o, s in np.ndindex(spatial_kernels.shape[:2]):
        assert np.allclose(spatial_kernels[o, s], RHS_kernels[o, s])


def test_normalized_outputs(model, MATLAB_filteroutput, params_FLODOG_MATLAB):
    weighted_outputs = RHS_implementation.weight(MATLAB_filteroutput)
    RHS_normalized_outputs = RHS_implementation.FLODOG_normalize(
        weighted_outputs,
        sigx=params_FLODOG_MATLAB["sigx"],
        sdmix=params_FLODOG_MATLAB["sdmix"],
    )

    normed_outputs = model.normalize_outputs(weighted_outputs, eps=1e-6)

    assert np.allclose(normed_outputs, RHS_normalized_outputs)


def test_model_output(output_FLODOG_MATLAB, model, stimulus):
    output = model.apply(stimulus, eps=1e-6)
    assert np.allclose(output, output_FLODOG_MATLAB)
