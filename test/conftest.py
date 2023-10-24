import os

import numpy as np
import pytest
import RHS_filters
from scipy import io

filepath_matlab_output = os.path.abspath(__file__ + "../../odog_matlab.mat")


@pytest.fixture()
def rhs_bank():
    # Create RHS filterbank from Python transplation
    return RHS_filters.filterbank()


@pytest.fixture()
def matlab_bank():
    # Load RHS bank from MATLAB implementation
    return np.array(io.loadmat(filepath_matlab_output)["filters"].tolist())


@pytest.fixture()
def stimulus():
    return io.loadmat(filepath_matlab_output)["illusion"]


@pytest.fixture()
def matlab_filteroutput():
    # Load RHS bank from MATLAB implementation
    return np.array(io.loadmat(filepath_matlab_output)["filter_response"].tolist())


@pytest.fixture()
def output_odog_matlab():
    return io.loadmat(filepath_matlab_output)["odog_output"]
