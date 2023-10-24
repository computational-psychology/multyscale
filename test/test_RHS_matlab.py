# %%
import matplotlib.pyplot as plt
import numpy as np
import RHS_filters


def test_filterbank(rhs_bank, matlab_bank):
    # %% Visualise
    for i in range(rhs_bank.shape[0]):
        plt.subplot(rhs_bank.shape[0], 2, i * 2 + 1)
        plt.imshow(rhs_bank[i, 6, ...])
        plt.subplot(rhs_bank.shape[0], 2, i * 2 + 2)
        plt.imshow(rhs_bank[i, 6, ...])

    assert np.allclose(rhs_bank, matlab_bank)


def test_RHSconv_matlab(matlab_filteroutput, matlab_bank, stimulus):
    # RHS convolution with matlab filters matches matlab output
    filters_output = np.empty(matlab_bank.shape)
    for i in range(matlab_bank.shape[0]):
        for j in range(matlab_bank.shape[1]):
            filters_output[i, j, ...] = RHS_filters.ourconv(stimulus, matlab_bank[i, j, ...])

    assert np.allclose(matlab_filteroutput, filters_output)


def test_RHSconv_RHS(matlab_filteroutput, stimulus, rhs_bank):
    # RHS convolution with python RHS filters matches matlab output
    filters_output = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            filters_output[i, j, ...] = RHS_filters.ourconv(stimulus, rhs_bank[i, j, ...])

    assert np.allclose(matlab_filteroutput, filters_output)


def test_ODOG(stimulus, rhs_bank, output_odog_matlab):
    # RHS convolution with python RHS filters matches matlab output
    filters_output = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            filters_output[i, j, ...] = RHS_filters.ourconv(stimulus, rhs_bank[i, j, ...])

    output = RHS_filters.odog_normalize(filters_output)
    assert np.allclose(output, output_odog_matlab)
