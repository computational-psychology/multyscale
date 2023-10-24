import matplotlib.pyplot as plt
import numpy as np
import RHS_implementation

from multyscale import filterbank

# %% Parameters of image
shape = (1024, 1024)  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
# NOTE: RHS implementation doesn't actually use (-16,16,-16,16)


# %% Filterbank
def test_filterbank(matlab_bank):
    multy_bank = filterbank.RHS2007((1024, 1024), visextent)
    #  Visualise filterbank
    for i in range(multy_bank.filters.shape[0]):
        plt.subplot(multy_bank.filters.shape[0], 2, i * 2 + 1)
        plt.imshow(multy_bank.filters[i, 6, ...], extent=visextent)
        plt.subplot(multy_bank.filters.shape[0], 2, i * 2 + 2)
        plt.imshow(matlab_bank[i, 6, ...])

    assert np.allclose(matlab_bank, multy_bank.filters)


def test_filterbank_apply(stimulus, matlab_filteroutput):
    multy_bank = filterbank.RHS2007((1024, 1024), visextent)
    multy_output = multy_bank.apply(stimulus)

    assert np.allclose(matlab_filteroutput, multy_output)


def test_scale_weights():
    bank = filterbank.RHS2007(shape, visextent)
    center_sigmas = np.array(bank.sigmas)[:, 0, 0]
    scale_weights = filterbank.scale_weights(center_sigmas, 0.1)
    assert np.allclose(scale_weights, RHS_implementation.w_val)
