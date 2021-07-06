# %%
import RHS_filters
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

import pytest
import os

current_dir = os.path.abspath(__file__ + "../../")


@pytest.fixture()
def rhs_bank():
    # Create RHS filterbank from Python transplation
    return RHS_filters.filterbank()


@pytest.fixture()
def matlab_bank():
    # Load RHS bank from MATLAB implementation
    return np.array(
        io.loadmat(current_dir + os.sep + "odog_matlab.mat")[
            "filters"
        ].tolist()
    )


@pytest.fixture()
def stimulus():
    return io.loadmat(current_dir + os.sep + "odog_matlab.mat")["illusion"]


@pytest.fixture()
def matlab_filteroutput():
    # Load RHS bank from MATLAB implementation
    return np.array(
        io.loadmat(current_dir + os.sep + "odog_matlab.mat")[
            "filter_response"
        ].tolist()
    )


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
            filters_output[i, j, ...] = RHS_filters.ourconv(
                stimulus, matlab_bank[i, j, ...]
            )

    assert np.allclose(matlab_filteroutput, filters_output)


def test_RHSconv_RHS(matlab_filteroutput, stimulus, rhs_bank):
    # RHS convolution with python RHS filters matches matlab output
    filters_output = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            filters_output[i, j, ...] = RHS_filters.ourconv(
                stimulus, rhs_bank[i, j, ...]
            )

    assert np.allclose(matlab_filteroutput, filters_output)
