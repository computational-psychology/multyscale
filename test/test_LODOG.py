# %% Imports
import numpy as np
import RHS_implementation

import multyscale

# %% Parameters of image
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
# NOTE: RHS implementation doesn't actually use (-16,16,-16,16)
window_sigma = 128 / 32


# %%
def test_lodog_filters(stimulus, MATLAB_filteroutput):
    model = multyscale.models.LODOG_RHS2007(stimulus.shape, visextent, window_sigma=window_sigma)

    lodog_filter_output = model.bank.apply(stimulus)

    assert np.allclose(MATLAB_filteroutput, lodog_filter_output)


def test_weights(stimulus):
    model = multyscale.models.LODOG_RHS2007(stimulus.shape, visextent, window_sigma=window_sigma)
    assert np.allclose(model.scale_weights, RHS_implementation.w_val)


def test_lodog_normalizers(stimulus, MATLAB_filteroutput):
    model = multyscale.models.LODOG_RHS2007(stimulus.shape, visextent, window_sigma=window_sigma)
    weighted_outputs = model.weight_outputs(MATLAB_filteroutput)
    normalizers = model.normalizers(weighted_outputs)
    RHS_norms = RHS_implementation.lodog_normalizers(weighted_outputs)
    assert np.allclose(normalizers.shape, RHS_norms.shape)
    assert np.allclose(normalizers, RHS_norms)


def test_lodog_mask(stimulus):
    # Is the spatial (Gaussian) averaging window the same?
    RHS_mask = RHS_implementation.lodog_mask(sig1=128)
    model = multyscale.models.LODOG_RHS2007(stimulus.shape, visextent, window_sigma=window_sigma)
    spatial_avg_filters = multyscale.normalization.spatial_avg_windows_gaussian(
        model.bank.x, model.bank.y, model.window_sigmas
    )
    for o, s in np.ndindex(spatial_avg_filters.shape[:2]):
        assert np.allclose(spatial_avg_filters[o, s], RHS_mask)


def test_lodog_RMS(stimulus):
    model = multyscale.models.LODOG_RHS2007(stimulus.shape, visextent, window_sigma=window_sigma)
    filter_outputs = model.bank.apply(stimulus)
    weighted_outputs = model.weight_outputs(filter_outputs)
    normalizers = model.normalizers(weighted_outputs)

    RMSs = model.normalizers_to_RMS(normalizers)
    RHS_RMSs = RHS_implementation.lodog_RMSs(normalizers, sig1=128)
    assert np.allclose(RMSs, RHS_RMSs)


def test_normalized_outputs(stimulus):
    model = multyscale.models.LODOG_RHS2007(stimulus.shape, visextent, window_sigma=window_sigma)
    filter_outputs = model.bank.apply(stimulus)
    weighted_outputs = model.weight_outputs(filter_outputs)

    normed_outputs = model.normalize_outputs(weighted_outputs)

    RHS_normalized_outputs = RHS_implementation.lodog_normalize(weighted_outputs, sig1=128)
    assert np.allclose(normed_outputs, RHS_normalized_outputs)


def test_lodog_output(stimulus, output_lodog_MATLAB):
    model = multyscale.models.LODOG_RHS2007(stimulus.shape, visextent, window_sigma=window_sigma)

    output = model.apply(stimulus)
    assert np.allclose(output, output_lodog_MATLAB)
