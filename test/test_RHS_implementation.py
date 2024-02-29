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


def test_filterbank(RHS_bank, MATLAB_bank):
    """Python filterbank matches RHS MATLAB filterbank"""
    assert np.allclose(RHS_bank, MATLAB_bank)


def test_RHSconv_MATLAB(MATLAB_filteroutput, MATLAB_bank, stimulus):
    """Python convolution with RHS MATLAB filters matches RHS MATLAB filters output"""
    filters_output = np.empty(MATLAB_bank.shape)
    for o, s in np.ndindex(MATLAB_bank.shape[:2]):
        filters_output[o, s, ...] = RHS_implementation.ourconv(
            stimulus, MATLAB_bank[o, s, ...], pad=0.5
        )

    assert np.allclose(MATLAB_filteroutput, filters_output)


def test_RHSconv_RHS(MATLAB_filteroutput, stimulus, RHS_bank):
    """Python convolution with Python filters matches RHS MATLAB filters output"""
    filters_output = np.empty(RHS_bank.shape)
    for o, s in np.ndindex(RHS_bank.shape[:2]):
        filters_output[o, s, ...] = RHS_implementation.ourconv(
            stimulus, RHS_bank[o, s, ...], pad=0.5
        )

    assert np.allclose(MATLAB_filteroutput, filters_output)


def test_ODOG(output_ODOG_MATLAB, MATLAB_filteroutput):
    """Python ODOG normalization & output matches RHS MATLAB ODOG output"""

    # Normalize and read out
    output = RHS_implementation.odog_normalize(MATLAB_filteroutput)

    # Compare
    assert np.allclose(output, output_ODOG_MATLAB)


def test_LODOG(output_LODOG_MATLAB, MATLAB_filteroutput, params_LODOG_MATLAB):
    """Python LODOG normalization & output matches RHS MATLAB LODOG output"""

    # Weight filteroutput by scale
    filters_output = RHS_implementation.weight(MATLAB_filteroutput)

    # Normalize
    normed_multi_responses = RHS_implementation.LODOG_normalize(
        filters_output,
        **params_LODOG_MATLAB,
    )

    # Readout model output by summing normalized channel outputs
    output = np.sum(normed_multi_responses, (0, 1))

    # Compare to MATLAB output
    assert np.allclose(output, output_LODOG_MATLAB)


def test_FLODOG(output_FLODOG_MATLAB, MATLAB_filteroutput, params_FLODOG_MATLAB):
    """Python FLODOG normalization & output matches RHS MATLAB FLODOG output"""

    # Weight filteroutput by scale
    filters_output = RHS_implementation.weight(MATLAB_filteroutput)

    # Normalize
    normed_multi_responses = RHS_implementation.FLODOG_normalize(
        filters_output,
        **params_FLODOG_MATLAB,
    )

    # Sum normalized channel outputs, to read out model output
    output = np.sum(normed_multi_responses, (0, 1))

    # Compare to MATLAB output
    assert np.allclose(output, output_FLODOG_MATLAB)
