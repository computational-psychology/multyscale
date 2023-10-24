# %%
import matplotlib.pyplot as plt
import numpy as np
import RHS_implementation

import multyscale

# %% RHS bank
rhs_bank = RHS_implementation.filterbank()

# %% Parameters of image
shape = (1024, 1024)  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)

# %% Create image coordinate system:
axish = np.linspace(visextent[0], visextent[1], shape[0])
axisv = np.linspace(visextent[2], visextent[3], shape[1])

(x, y) = np.meshgrid(axish, axisv)


# %% Circular Gaussian
def test_circular_Gaussian():
    sigma1 = 2
    sigmas = np.array([1, 1]) * sigma1
    f = multyscale.filters.gaussian2d(x, y, (sigmas[0], sigmas[1]))
    f = f / f.sum()
    f_2 = RHS_implementation.d2gauss(shape[0], sigmas[0] * 32, shape[1], sigmas[0] * 32, 0)

    plt.subplot(2, 2, 1)
    plt.imshow(f)
    plt.subplot(2, 2, 2)
    plt.imshow(f_2)
    plt.subplot(2, 2, 3)
    plt.plot(f[512, :])
    plt.subplot(2, 2, 4)
    plt.plot(f_2[512, :])

    assert np.allclose(f, f_2)


# %% Elliptical Gaussian
def test_elliptical_Gaussian():
    sigma1 = 2
    orientation = 40
    sigma2 = 2 * sigma1
    sigmas = np.array([1, 1]) * np.array([sigma1, sigma2])
    f = multyscale.filters.gaussian2d(x, y, (sigmas[0], sigmas[1]), orientation=orientation)
    f = f / f.sum()
    f_2 = RHS_implementation.d2gauss(
        shape[0], sigmas[0] * 32, shape[1], sigmas[1] * 32, orientation
    )

    plt.subplot(2, 2, 1)
    plt.imshow(f)
    plt.subplot(2, 2, 2)
    plt.imshow(f_2)
    plt.subplot(2, 2, 3)
    plt.plot(f[512, :])
    plt.subplot(2, 2, 4)
    plt.plot(f_2[512, :])

    assert np.allclose(f, f_2)


# %% ODOG
def test_ODOG():
    orientation = 150
    sigma3 = 2
    sigmas = np.array([[1, 1], [1, 2]]) * sigma3
    rhs_odog = RHS_implementation.odog(shape[0], shape[1], sigma3 * 32, orientation=orientation)
    multy_odog = multyscale.filters.odog(x, y, sigmas, orientation=(orientation, orientation))

    plt.subplot(2, 2, 1)
    plt.imshow(rhs_odog)
    plt.subplot(2, 2, 2)
    plt.imshow(multy_odog)
    plt.subplot(2, 2, 3)
    plt.plot(rhs_odog[512, :])
    plt.subplot(2, 2, 4)
    plt.plot(multy_odog[512, :])

    assert np.allclose(rhs_odog, multy_odog)
