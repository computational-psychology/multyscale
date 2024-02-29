from pathlib import Path

import numpy as np
import pytest
from scipy import io

filepath_MATLAB = Path(__file__).parent / "MATLAB_RHS2007_(F)(L)ODOG.mat"


@pytest.fixture()
def MATLAB_visextent():
    # Visual extent, same convention as pyplot (Left, Right, Bottom, Top):
    visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)
    # NOTE: RHS implementation doesn't actually use (-16,16,-16,16)
    return visextent


@pytest.fixture()
def MATLAB_shape(MATLAB_bank):
    # Shape (resolution) of image, filters (Y, X)
    return MATLAB_bank.shape[-2:]


@pytest.fixture()
def MATLAB_bank():
    # Load RHS bank from MATLAB implementation
    return np.array(io.loadmat(filepath_MATLAB)["filterbank"].tolist())


@pytest.fixture()
def stimulus():
    return io.loadmat(filepath_MATLAB)["stimulus"]


@pytest.fixture()
def MATLAB_filteroutput():
    # Load RHS bank from MATLAB implementation
    return np.array(io.loadmat(filepath_MATLAB)["filters_output"].tolist())


@pytest.fixture()
def output_ODOG_MATLAB():
    return io.loadmat(filepath_MATLAB)["output_ODOG"]


@pytest.fixture()
def output_LODOG_MATLAB():
    return io.loadmat(filepath_MATLAB)["output_LODOG"]


@pytest.fixture()
def output_FLODOG_MATLAB():
    return io.loadmat(filepath_MATLAB)["output_FLODOG"]


@pytest.fixture()
def params_LODOG_MATLAB():
    """Normalization parameters used to produce the MATLAB output for LODOG"""
    params_LODOG = io.loadmat(filepath_MATLAB)["params_LODOG"]
    params_LODOG = {name: float(params_LODOG[name]) for name in params_LODOG.dtype.names}
    return params_LODOG


@pytest.fixture()
def params_FLODOG_MATLAB():
    """Normalization parameters used to produce the MATLAB output for FLODOG"""
    params_FLODOG = io.loadmat(filepath_MATLAB)["params_FLODOG"]
    params_FLODOG = {name: float(params_FLODOG[name]) for name in params_FLODOG.dtype.names}
    return params_FLODOG
