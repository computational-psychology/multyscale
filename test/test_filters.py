import numpy as np
import RHS_implementation

import multyscale

# %% RHS bank
RHS_bank = RHS_implementation.filterbank()

# %% Parameters of image
shape = (1024, 1024)  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)

# %% Create image coordinate system:
axish = np.linspace(visextent[0], visextent[1], shape[0])
axisv = np.linspace(visextent[2], visextent[3], shape[1])

(x, y) = np.meshgrid(axish, axisv)


def test_circular_Gaussian():
    """Test that multyscale's (circular) Gaussian is equal to RHS (circular) Gaussian"""

    # Filter params
    sigma1 = 2
    sigmas = np.array([1, 1]) * sigma1

    # Generate filter
    filter_multy = multyscale.filters.gaussian2d(x, y, (sigmas[0], sigmas[1]))
    filter_multy = filter_multy / filter_multy.sum()

    # Generate comparison filter
    filter_multy = RHS_implementation.d2gauss(
        shape[0], sigmas[0] * 32, shape[1], sigmas[0] * 32, 0
    )

    assert np.allclose(filter_multy, filter_multy)


def test_elliptical_Gaussian():
    """Test that multyscale's (elliptical) Gaussian is equal to RHS (elliptical) Gaussian"""

    # Filter params
    sigma1 = 2
    orientation = 40
    sigma2 = 2 * sigma1
    sigmas = np.array([1, 1]) * np.array([sigma1, sigma2])

    # Generate filter
    filter_multy = multyscale.filters.gaussian2d(
        x, y, (sigmas[0], sigmas[1]), orientation=orientation
    )
    filter_multy = filter_multy / filter_multy.sum()

    # Generate comparison filter
    filter_RHS = RHS_implementation.d2gauss(
        shape[0], sigmas[0] * 32, shape[1], sigmas[1] * 32, orientation
    )

    assert np.allclose(filter_multy, filter_RHS)


def test_ODOG():
    """Test that multyscale's ODOG filter is equal to RHS ODOG filter"""

    # Filter params
    orientation = 150
    sigma3 = 2
    sigmas = np.array([[1, 1], [1, 2]]) * sigma3

    # Generate filter
    filter_RHS = RHS_implementation.odog(shape[0], shape[1], sigma3 * 32, orientation=orientation)

    # Generate comparison filter
    filter_multy = multyscale.filters.odog(x, y, sigmas, orientation=(orientation, orientation))

    assert np.allclose(filter_RHS, filter_multy)
