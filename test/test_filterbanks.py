import numpy as np
import matplotlib.pyplot as plt

from multyscale import filterbank

# %% Parameters of image
shape = (1024, 1024)  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
# NOTE: this is NOT (-16,16,-16,16), because RHS implementation doesn't actually use that


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
