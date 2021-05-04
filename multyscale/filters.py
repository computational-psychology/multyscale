# Third party imports
import numpy as np
from scipy import signal

# TODO: (Abstract) base class Filter with apply-method,...


def apply(image, filt):
    # TODO: docstring
    # TODO: make method
    filtered_image = signal.fftconvolve(image, filt, mode="same")
    return filtered_image


def gaussian2d(x, y, sigma, center=(0, 0), orientation=0):
    # TODO: convert Gaussian to class
    # TODO: typehints
    # TODO: docstring

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
    # TODO: convert ODoG to class
    # TODO: typehints
    # TODO: docstring

    # Sigma is tuple of two tuples, one for Gaussian.
    # Orientation is tuple of two floats, one for each

    # Create center and surround anisotropic Gaussian filters
    center = gaussian2d(x, y, sigma=sigma[0], orientation=orientation[0])
    surround = gaussian2d(x, y, sigma=sigma[1], orientation=orientation[1])

    # Normalize each filter by its total
    center = center / center.sum()
    surround = surround / surround.sum()

    # Subtract to create differential filter
    odog = center - surround

    return odog


def dog(x, y, sigma):
    # TODO: docstring
    # TODO: typehints

    # Isotropic difference of Gaussian;
    # Sigma is tuple of two floats, one for each Gaussian.
    # center has (sigma[0], sigma[0]),
    # surround has (sigma[1], sigma[1])
    dog = odog(x, y, ((sigma[0], sigma[0]), (sigma[1], sigma[1])))

    return dog
