import matplotlib.pyplot as plt
import numpy as np
import RHS_implementation

from multyscale import filterbanks

# %% Parameters of image
shape = (1024, 1024)  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
# NOTE: RHS implementation doesn't actually use (-16,16,-16,16)


def test_filterbank(MATLAB_bank):
    multy_bank = filterbanks.RHS2007((1024, 1024), visextent)

    assert np.allclose(MATLAB_bank, multy_bank.filters)


def test_filterbank_apply(stimulus, MATLAB_filteroutput):
    multy_bank = filterbanks.RHS2007((1024, 1024), visextent)
    multy_output = multy_bank.apply(stimulus)

    assert np.allclose(MATLAB_filteroutput, multy_output)


def test_scale_weights():
    bank = filterbanks.RHS2007(shape, visextent)
    center_sigmas = np.array(bank.sigmas)[:, 0, 0]
    scale_weights = filterbanks.scale_weights(center_sigmas, 0.1)
    assert np.allclose(scale_weights, RHS_implementation.w_val)
