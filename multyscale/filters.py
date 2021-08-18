# Python Standard Library
from __future__ import annotations
from collections.abc import Sequence

# Third party imports
import numpy as np
from scipy import signal

# TODO: (Abstract) base class Filter with apply-method,...


def apply(
    image: np.ndarray, filt: np.ndarray, pad: bool = False
) -> np.ndarray:
    """Apply filter to image, optionally pad input

    Parameters
    ----------
    image : numpy.ndarray
        image to filter
    filt : numpy.ndarray
        filter to use
    pad : bool, optional
        whether to pad the input image, by default False.
        If true, input will be padded equally around all borders
        with the constant value 0.5
        to be the size of the filter,

    Returns
    -------
    numpy.ndarray
        Filtered image
    """

    # TODO: make method
    if pad:
        pad_vertical, pad_horizontal = np.array(filt.shape)
        padding = np.array(
            [
                [pad_vertical / 2 - 1, pad_vertical / 2],
                [pad_horizontal / 2 - 1, pad_horizontal / 2],
            ],
            dtype="int",
        )
        pad_image = np.pad(image, padding, "constant", constant_values=0.5)
        filtered_image = signal.fftconvolve(pad_image, filt, mode="valid")
    else:
        filtered_image = signal.fftconvolve(image, filt, mode="same")
    return filtered_image


def gaussian2d(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Sequence[float],
    center: Sequence[int] = (0, 0),
    orientation: float = 0,
) -> np.ndarray:
    """Create 2D Gaussian

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates of each pixel
    y : numpy.ndarray
        y coordinates of each pixel
    sigma : Sequence[float]
        standard deviation along each axis (2-vector) or both axes (scalar)
    center : Sequence[int], optional
        coordinates to center Gaussian on, by default (0, 0)
    orientation : float, optional
        degrees of counterclockwise rotation of Gaussian axes, by default 0

    Returns
    -------
    numpy.ndarray
        image matrix, of shape = x.shape, with a Gaussian kernel
    """
    # TODO: convert Gaussian to class

    # Sigma is tuple of two sigmas, one for each axis.
    # Units of x,y, determine units of sigma:
    # if x,y are in pixels, e.g., [0,1024],
    # sigma should be specified in pixels as well;
    # if x,y are in degress, e.g., [-16, 16],
    # sigma should be specified in degrees as well.

    # General equation for a 2D elliptical Gaussian:
    #
    # f(x,y) = A*exp(-(a(x-x0)^2+2b(x-x0)(y-y0)+c(y-y0)2))
    #
    # where x0 and x0 are the center positions.
    # Let's leave amplitude A as responsibility to the caller,
    # and keep it fixed at 1 here.
    # If center = (0,0), this reduces the equation to:
    #
    # f(x,y) = exp(-(ax^2+2bxy+cy^2))
    #
    # where, given angle th in radians:
    #
    # a = (cos(th)^2 / 2sigma_x^2) + (sin(th)^2 / 2sigma_y^2)
    # b = -(sin(2*th)/ 4sigma_x^2) + (sin(2*th) / 4sigma_y^2)
    # c = (sin(th)^2 / 2sigma_x^2) + (cos(th)^2 / 2sigma_y^2)

    # convert orientation parameter to radians
    sigma = np.array(sigma) * [1, 1]

    theta = np.deg2rad(orientation)

    # determine a, b, c coefficients
    a = (np.cos(theta) ** 2 / (2 * sigma[0] ** 2)) + (
        np.sin(theta) ** 2 / (2 * sigma[1] ** 2)
    )
    b = -(np.sin(2 * theta) / (4 * sigma[0] ** 2)) + (
        np.sin(2 * theta) / (4 * sigma[1] ** 2)
    )
    c = (np.sin(theta) ** 2 / (2 * sigma[0] ** 2)) + (
        np.cos(theta) ** 2 / (2 * sigma[1] ** 2)
    )

    # create Gaussian
    gaussian = np.exp(
        -(
            a * (x - center[0]) ** 2
            + 2 * b * (x - center[0]) * (y - center[1])
            + c * (y - center[1]) ** 2
        )
    )

    return gaussian


def odog(x, y, sigma, orientation=(0, 0)):
    """Create oriented difference-of-Gaussian filter

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates of each pixel
    y : numpy.ndarray
        y coordinates of each pixel
    sigma : Sequence[Sequence[float]]
        standard deviation along each axis (2-vector) or both axes (scalar)
        for center Gaussian,
        and similar for surround Gaussian
    center : Sequence[int], optional
        coordinates to center Gaussian on, by default (0, 0)
    orientation : Sequence[float], optional
        degrees of counterclockwise rotation of axes of each Gaussian, by default (0, 0)

    Returns
    -------
    numpy.ndarray
        image matrix, of shape = x.shape, with oriented DoG kernel
    """

    # TODO: convert ODoG to class

    # Sigma is tuple of two tuples, one for Gaussian.
    # Orientation is tuple of two floats, one for each

    orientation = np.array(orientation) * [1, 1]

    # Create center and surround anisotropic Gaussian filters
    center = gaussian2d(x, y, sigma=sigma[0], orientation=orientation[0])
    surround = gaussian2d(x, y, sigma=sigma[1], orientation=orientation[1])

    # Normalize each filter by its total
    center = center / center.sum()
    surround = surround / surround.sum()

    # Subtract to create differential filter
    odog = center - surround

    return odog


def dog(
    x: np.ndarray,
    y: np.ndarray,
    sigma: Sequence[float],
    center: Sequence[int] = (0, 0),
) -> np.ndarray:
    """Create isotropic (unoriented) difference-of-Gaussian filter

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates of each pixel
    y : numpy.ndarray
        y coordinates of each pixel
    sigma : Sequence[float]
        standard deviation along both axes (scalar)
        for center Gaussian, and for surround Gaussian
    center : Sequence[int], optional
        coordinates to center Gaussian on, by default (0, 0)

    Returns
    -------
    numpy.ndarray
        image matrix, of shape = x.shape, with isotropic DoG kernel
    """

    # Isotropic difference of Gaussian;
    # Sigma is tuple of two floats, one for each Gaussian.
    # center has (sigma[0], sigma[0]),
    # surround has (sigma[1], sigma[1])
    dog = odog(x, y, ((sigma[0], sigma[0]), (sigma[1], sigma[1])))

    return dog


def global_avg(shape: Sequence[int]) -> np.ndarray:
    """Create filter to calculate global mean

    Parameters
    ----------
    shape : Sequence[int]
        Size (h,w), in pixels of filter

    Returns
    -------
    np.ndarray
        Image matrix defining filter, where each entry is 1/(h * w).
    """
    filt_avg = np.ones(shape)
    filt_avg = filt_avg / filt_avg.sum(axis=(-1, -2))
    return filt_avg
