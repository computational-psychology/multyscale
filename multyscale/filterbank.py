"""Structured sets of different types of filters, and functions for weighting outputs

Primarily classes for banks of different filter types,
but also includes convience functions to produce banks of filters
that correspond to Blakeslee & McCourt (1997; 1999)
and Robinson, Hammon, & de Sa (2007) filters.

"""


# Third party imports
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


# Local application imports
from . import filters, utils

# TODO: (abstract) base class Filterbank, with apply method

# TODO: data structure for filter output, with filter metadata


# Weighting filter(outputs) according to spatial scale
# Defining the scale weights
def scale_weights(center_sigmas, slope):
    # TODO: docstring
    scale_weights = (1 / np.asarray(center_sigmas)) ** slope
    return scale_weights


# Weighting multiscale output
def weight_multiscale_outputs(multiscale_filters, scale_weights):
    # TODO: docstring
    # Weight multiscale filters outputs according to filter spatial scale
    weighted_multiscale = [
        f * w for f, w in zip(multiscale_filters, scale_weights)
    ]
    return np.asarray(weighted_multiscale)


# Weighting oriented multiscales
def weight_oriented_multiscale_outputs(filters_output, scale_weights):
    # TODO: docstring
    # Weight each filter output according to filter spatial scale
    weighted_filters_output = [
        weight_multiscale_outputs(m, scale_weights) for m in filters_output
    ]
    return np.asarray(weighted_filters_output)


class DOGBank:
    """Bank of istropic (unoriented; symmetrical) difference-of-Gaussian filters
    Bank has a set of standard deviations of the filters

    Parameters
    ----------
    sigmas : array-like[array-like[float,float]]
        Standard deviations, two (center, surround) for each filter
    x : numpy.ndarray
        image grid of x coordinates for each pixel; easily created with meshgrid
    y : numpy.ndarray
        image grid of y coordinates for each pixel; easily created with meshgrid
    """

    def __init__(
        self, sigmas: Sequence[Sequence[float]], x: np.ndarray, y: np.ndarray
    ):
        self.sigmas = sigmas
        self.x = x
        self.y = y

        self.filters = np.empty((len(sigmas), x.shape[0], x.shape[1]))
        for i, sigma in enumerate(sigmas):
            dog = filters.dog(x, y, sigma)
            self.filters[i, :, :] = dog

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply filterbank to given image

        Parameters
        ----------
        image : numpy.ndarray
            Image matrix to be filtered

        Returns
        -------
        numpy.ndarray
            Multidimensional (N,X,Y) array of filter outputs, where
            N is the number of spatial scales in filterbank,
            X, Y are the (pixel)shape of the input image
        """
        filters_output = np.empty(self.filters.shape)
        for i in range(self.filters.shape[0]):
            filters_output[i, ...] = filters.apply(
                image, self.filters[i, ...], pad=True
            )
        return filters_output


def BM1997(
    filtershape: Sequence[int] = (1024, 1024),
    visextent: Sequence[float] = (-16, 16, -16, 16),
) -> DOGBank:
    """Bank of isotropic DoG filters with sizes used by Blakeslee & McCourt (1999)

    Parameters
    ----------
    filtershape : array-like[int], optional
        [description], by default (1024, 1024)
    visextent : array-like[float], optional
        [description], by default (-16, 16, -16, 16)

    Returns
    -------
    DOGBank
        [description]
    """
    # Parameters (BM1997)
    num_scales = 7
    largest_center_sigma = 3  # in degrees
    center_sigmas = utils.octave_intervals(num_scales) * largest_center_sigma
    cs_ratio = 2  # center-surround ratio

    # Convert to filterbank parameters
    sigmas = [(s, cs_ratio * s) for s in center_sigmas]

    # Create image coordinate system:
    axish = np.linspace(visextent[0], visextent[1], filtershape[0])
    axisv = np.linspace(visextent[2], visextent[3], filtershape[1])
    (x, y) = np.meshgrid(axish, axisv)

    # Create filterbank
    bank = DOGBank(sigmas, x, y)
    return bank


class ODOGBank:
    """Bank of oriented difference-of-Gaussian filters
    Bank has a set of orientations, a set of standard deviations of the filters

    Parameters
    ----------
    orientations : array-like[float]
        Array-like of orientations, one per filter, in degrees
    sigmas : array-like[array-like[[float, float],[float, float]]]]
        Array-like of standard deviations,
        two (center, surround) sets of
        two (major, minor axis) per filter
    x : numpy.ndarray
        image grid of x coordinates for each pixel; easily created with meshgrid
    y : numpy.ndarray
        image grid of y coordinates for each pixel; easily created with meshgrid
    """

    def __init__(
        self,
        orientations: Sequence[float],
        sigmas: Sequence[Sequence[Sequence[float]]],
        x: np.ndarray,
        y: np.ndarray,
    ):

        # TODO: don't store filtes as ND-array
        self.orientations = orientations
        self.sigmas = sigmas
        self.x = x
        self.y = y

        self.filters = np.empty(
            (len(orientations), len(sigmas), x.shape[0], x.shape[1])
        )

        for i, angle in enumerate(orientations):
            for j, sigma in enumerate(sigmas):
                odog = filters.odog(x, y, sigma, (angle, angle))
                self.filters[i, j, :, :] = odog

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply filterbank to given image

        Parameters
        ----------
        image : numpy.ndarray
            Image matrix to be filtered

        Returns
        -------
        numpy.ndarray
            Multidimensional (M,N,X,Y) array of filter outputs, where
            M is the number of orientations in filterbank,
            N is the number of spatial scales in filterbank,
            X, Y are the (pixel)shape of the input image
        """
        filters_output = np.empty(self.filters.shape)
        for i in range(self.filters.shape[0]):
            for j in range(self.filters.shape[1]):
                filters_output[i, j, ...] = filters.apply(
                    image, self.filters[i, j, ...], pad=True
                )
        return filters_output


def BM1999(
    filtershape: Sequence[int] = (1024, 1024),
    visextent: Sequence[float] = (-16, 16, -16, 16),
) -> ODOGBank:
    """ODOG bank with orientations and spatial scales as used by BM1999 and RHS2007

    Parameters
    ----------
    filtershape : array-like[int], optional
        Pixel size (W,H) of filters, by default (1024, 1024)
    visextent : array-like[float], optional
        Extent in degrees visual angle, (left, right, top, bottom),
        by default (-16, 16, -16, 16)

    Returns
    -------
    ODOGBank
        Bank of ODOG filters as used by BM1999 and RHS2007, with
        6 orientations: [0, 30, 60, 90, 120, 150] degrees
        7 spatial scales: octave intervals, down from 3 degrees visual angle
    """
    # TODO: figure out actual BM space parameter....

    # Parameters (BM1999)
    n_orientations = 6
    num_scales = 7
    largest_center_sigma = 3  # in degrees
    center_sigmas = utils.octave_intervals(num_scales) * largest_center_sigma
    cs_ratio = 2  # center-surround ratio

    # Convert to filterbank parameters
    orientations = np.arange(0, 180, 180 / n_orientations)
    sigmas = [((s, s), (s, cs_ratio * s)) for s in center_sigmas]

    # Create image coordinate system:
    axish = np.linspace(visextent[0], visextent[1], filtershape[0])
    axisv = np.linspace(visextent[2], visextent[3], filtershape[1])
    (x, y) = np.meshgrid(axish, axisv)

    # Create filterbank
    bank = ODOGBank(orientations, sigmas, x, y)
    return bank
