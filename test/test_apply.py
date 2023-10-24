import numpy as np
import RHS_implementation

from multyscale import filters


def test_apply_MATLAB(MATLAB_filteroutput, MATLAB_bank, stimulus):
    # multyscale apply with MATLAB filters matches MATLAB output
    filters_output = np.empty(MATLAB_bank.shape)
    for i in range(MATLAB_bank.shape[0]):
        for j in range(MATLAB_bank.shape[1]):
            filters_output[i, j, ...] = filters.apply(stimulus, MATLAB_bank[i, j, ...], pad=True)

    assert np.allclose(MATLAB_filteroutput, filters_output)


def test_apply_RHS(MATLAB_filteroutput, rhs_bank, stimulus):
    # multyscale apply with RHS filters matches MATLAB output
    filters_output = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            filters_output[i, j, ...] = filters.apply(stimulus, rhs_bank[i, j, ...], pad=True)

    assert np.allclose(MATLAB_filteroutput, filters_output)


def test_conv_apply(rhs_bank, stimulus):
    o_conv = np.empty(rhs_bank.shape)
    o_apply = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            f = rhs_bank[i, j]
            o_conv[i, j, ...] = RHS_implementation.ourconv(stimulus, f)
            o_apply[i, j, ...] = filters.apply(stimulus, f, pad=True)

    assert np.allclose(o_conv, o_apply)
