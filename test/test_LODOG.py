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


def test_norm_coeffs(model, MATLAB_filteroutput):
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)
    norm_coeffs = model.norm_coeffs(weighted_outputs)
    RHS_norms = RHS_implementation.LODOG_normalizers(weighted_outputs)
    assert np.allclose(norm_coeffs.shape, RHS_norms.shape)
    assert np.allclose(norm_coeffs, RHS_norms)


def test_spatial_mask(model, MATLAB_LODOG_params):
    # Is the spatial (Gaussian) averaging window the same?
    RHS_kernel = RHS_implementation.LODOG_mask(sig1=MATLAB_LODOG_params["sig1"])
    spatial_kernels = model.spatial_kernels()
    for o, s in np.ndindex(spatial_kernels.shape[:2]):
        assert np.allclose(spatial_kernels[o, s], RHS_kernel)


def test_normalized_outputs(model, MATLAB_filteroutput, MATLAB_LODOG_params):
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)

    normed_outputs = model.normalize_outputs(weighted_outputs, eps=1e-6)

    RHS_normalized_outputs = RHS_implementation.LODOG_normalize(
        weighted_outputs, sig1=MATLAB_LODOG_params["sig1"]
    )
    assert np.allclose(normed_outputs, RHS_normalized_outputs)


def test_model_output(output_LODOG_MATLAB, model, stimulus):
    output = model.apply(stimulus, eps=1e-6)
    assert np.allclose(output, output_LODOG_MATLAB)
