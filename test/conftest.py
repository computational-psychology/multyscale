import pytest
import RHS_implementation
from MATLAB_comparison import *


@pytest.fixture()
def RHS_bank():
    # Create RHS filterbank from Python transplation
    return RHS_implementation.filterbank()
