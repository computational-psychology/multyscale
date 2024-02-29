import numpy as np
import pytest
import RHS_implementation

import multyscale

# %% Global parametrization
# Parameters of image
shape = (1024, 1024)  # in pixels
PPD = 32
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / PPD)

# Image coordinate system
axish = np.linspace(visextent[0], visextent[1], shape[0])
axisv = np.linspace(visextent[2], visextent[3], shape[1])
(x, y) = np.meshgrid(axish, axisv)

pytestmark = [pytest.mark.parametrize("sigma_center", RHS_implementation.std)]


# %% TEST
@pytest.mark.parametrize(
    "sigma_ratio, orientation",
    [
        (1, 0),  # circular, no need for orientation
        (2, 0),  # elliptical, cycle through all orientations
        (2, 30),
        (2, 60),
        (2, 90),
        (2, 120),
        (2, 150),
    ],
)
def test_Gaussian(sigma_ratio, sigma_center, orientation):
    """Test that multyscale's Gaussian is equal to RHS Gaussian"""

    # Filter params
    sigmas = np.array([1, sigma_ratio]) * sigma_center

    # Generate filter
    filter_multy = multyscale.filters.gaussian2d(
        x, y, (sigmas[0], sigmas[1]), orientation=orientation
    )
    filter_multy /= filter_multy.sum()

    # Generate comparison filter
    filter_multy = RHS_implementation.d2gauss(
        shape[0], sigmas[0] * PPD, shape[1], sigmas[0] * PPD, orientation
    )

    assert np.array_equal(filter_multy, filter_multy)


@pytest.mark.parametrize("sigma_ratio", [2])
@pytest.mark.parametrize("orientation", [0, 30, 60, 90, 120, 150])
def test_ODOG(sigma_center, sigma_ratio, orientation):
    """Test that multyscale's ODOG filter is equal to RHS ODOG filter"""

    # Filter params
    sigmas = np.array([[1, 1], [1, sigma_ratio]]) * sigma_center

    # Generate filter
    filter_RHS = RHS_implementation.odog(
        shape[0], shape[1], sigma_center * PPD, orientation=orientation
    )

    # Generate comparison filter
    filter_multy = multyscale.filters.odog(x, y, sigmas, orientation=(orientation, orientation))

    assert np.allclose(filter_RHS, filter_multy)
