from pathlib import Path

import numpy as np
import pytest
from scipy import io

filepath_MATLAB_output = Path(__file__).parent / "MATLAB_RHS2007_(F)(L)ODOG.mat"


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
    return np.array(io.loadmat(filepath_MATLAB_output)["filterbank"].tolist())


@pytest.fixture()
def stimulus():
    return io.loadmat(filepath_MATLAB_output)["stimulus"]


@pytest.fixture()
def MATLAB_filteroutput():
    # Load RHS bank from MATLAB implementation
    return np.array(io.loadmat(filepath_MATLAB_output)["filters_output"].tolist())


@pytest.fixture()
def output_ODOG_MATLAB():
    return io.loadmat(filepath_MATLAB_output)["output_ODOG"]


@pytest.fixture()
def output_LODOG_MATLAB():
    return io.loadmat(filepath_MATLAB_output)["output_LODOG"]


@pytest.fixture()
def output_FLODOG_MATLAB():
    return io.loadmat(filepath_MATLAB_output)["output_FLODOG"]


@pytest.fixture()
def MATLAB_LODOG_params():
    """Normalization parameters used to produce the MATLAB output for LODOG"""
    LODOG_params = {"sig1": 128, "sr": 1}
    return LODOG_params


@pytest.fixture()
def MATLAB_FLODOG_params():
    """Normalization parameters used to produce the MATLAB output for FLODOG"""
    FLODOG_params = {"sigx": 4, "sr": 1, "sdmix": 0.5}
    return FLODOG_params
