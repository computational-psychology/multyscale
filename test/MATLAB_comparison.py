import os

import numpy as np
import pytest
from scipy import io

filepath_MATLAB_output = os.path.abspath(__file__ + "../../odog_MATLAB.mat")


@pytest.fixture()
def MATLAB_visextent():
    # Visual extent, same convention as pyplot (Left, Right, Bottom, Top):
    visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
    # NOTE: RHS implementation doesn't actually use (-16,16,-16,16)
    return visextent


@pytest.fixture()
def MATLAB_shape():
    # Shape (resolution) of image, filters (Y, X)
    return (1024, 1024)


@pytest.fixture()
def MATLAB_bank():
    # Load RHS bank from MATLAB implementation
    return np.array(io.loadmat(filepath_MATLAB_output)["filters"].tolist())


@pytest.fixture()
def stimulus():
    return io.loadmat(filepath_MATLAB_output)["illusion"]


@pytest.fixture()
def MATLAB_filteroutput():
    # Load RHS bank from MATLAB implementation
    return np.array(io.loadmat(filepath_MATLAB_output)["filter_response"].tolist())


@pytest.fixture()
def output_ODOG_MATLAB():
    return io.loadmat(filepath_MATLAB_output)["odog_output"]


@pytest.fixture()
def output_lodog_MATLAB():
    return io.loadmat(filepath_MATLAB_output)["lodog_output"]


@pytest.fixture()
def output_flodog_MATLAB():
    return io.loadmat(filepath_MATLAB_output)["flodog_output"]
