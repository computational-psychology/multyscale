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
def rhs_matlab_bank():
    # Load RHS bank from MATLAB implementation
    return np.array(
        io.loadmat(current_dir + os.sep + "odog_matlab.mat")[
            "filters"
        ].tolist()
    )


def test_RHS_matlab(rhs_bank, rhs_matlab_bank):
    # %% Visualise
    for i in range(rhs_bank.shape[0]):
        plt.subplot(rhs_bank.shape[0], 2, i * 2 + 1)
        plt.imshow(rhs_bank[i, 6, ...])
        plt.subplot(rhs_bank.shape[0], 2, i * 2 + 2)
        plt.imshow(rhs_bank[i, 6, ...])

    assert np.allclose(rhs_bank, rhs_bank)
