import numpy as np
import RHS_filters

from multyscale import filters


def test_apply_matlab(matlab_filteroutput, matlab_bank, stimulus):
    # multyscale apply with matlab filters matches matlab output
    filters_output = np.empty(matlab_bank.shape)
    for i in range(matlab_bank.shape[0]):
        for j in range(matlab_bank.shape[1]):
            filters_output[i, j, ...] = filters.apply(stimulus, matlab_bank[i, j, ...], pad=True)

    assert np.allclose(matlab_filteroutput, filters_output)


def test_apply_RHS(matlab_filteroutput, rhs_bank, stimulus):
    # multyscale apply with RHS filters matches matlab output
    filters_output = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            filters_output[i, j, ...] = filters.apply(stimulus, rhs_bank[i, j, ...], pad=True)

    assert np.allclose(matlab_filteroutput, filters_output)


def test_conv_apply(rhs_bank, stimulus):
    o_conv = np.empty(rhs_bank.shape)
    o_apply = np.empty(rhs_bank.shape)
    for i in range(rhs_bank.shape[0]):
        for j in range(rhs_bank.shape[1]):
            f = rhs_bank[i, j]
            o_conv[i, j, ...] = RHS_filters.ourconv(stimulus, f)
            o_apply[i, j, ...] = filters.apply(stimulus, f, pad=True)

    assert np.allclose(o_conv, o_apply)
