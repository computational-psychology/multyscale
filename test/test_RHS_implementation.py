""" Test that Python transplementation matches original MATLAB output

The original Robinson, Hammon, de Sa (2007) implementation of (F)(L)ODOG is in MATLAB.
The multyscale testsuite provides a Python "transplementation" of the algorithms of this
original MATLAB code.

This module tests that the model output (and some in between steps) produced by this
Python transplementation matches numerically with the comparable output from original
MATLAB implementation

The "ground-truth" MATLAB output should be provided in file output_MATLAB.mat,
which is accessed by the pytest fixtures in MATLAB_comparison.py.

Note that this is selected output, for just a single stimulus.
"""

import numpy as np
import RHS_implementation


def test_filterbank(rhs_bank, matlab_bank):
    """Python filterbank matches RHS MATLAB filterbank"""
    assert np.allclose(rhs_bank, matlab_bank)


def test_RHSconv_matlab(matlab_filteroutput, matlab_bank, stimulus):
    """Python convolution with RHS MATLAB filters matches RHS MATLAB filters output"""
    filters_output = np.empty(matlab_bank.shape)
    for i in range(matlab_bank.shape[0]):
        for j in range(matlab_bank.shape[1]):
            filters_output[i, j, ...] = RHS_implementation.ourconv(
                stimulus, matlab_bank[i, j, ...]
            )

    assert np.allclose(matlab_filteroutput, filters_output)


def test_RHSconv_RHS(matlab_filteroutput, stimulus, rhs_bank):
    """Python convolution with Python filters matches RHS MATLAB filters output"""
    filters_output = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            filters_output[i, j, ...] = RHS_implementation.ourconv(stimulus, rhs_bank[i, j, ...])

    assert np.allclose(matlab_filteroutput, filters_output)


def test_ODOG(stimulus, rhs_bank, output_odog_matlab):
    """Python ODOG normalization & output matches RHS MATLAB ODOG output"""
    filters_output = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            filters_output[i, j, ...] = RHS_implementation.ourconv(stimulus, rhs_bank[i, j, ...])

    output = RHS_implementation.odog_normalize(filters_output)
    assert np.allclose(output, output_odog_matlab)
