# %% [markdown]
# # Creating and applying image filters

# %% Setup
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np

# Import local module
from multyscale import filters

# %% [markdown]
# ## Example stimulus
# The example stimulus used for this exploration
# is a version of White's (1979) classic illusion,
# as also used by Robinson, Hammon, & de Sa (2007) as `WE_thick`.
#
# This stimulus is provided here
# as an NumPy `.npy` file,
# so it can be loaded in directly as a NumPy ndarray.
#
# The image of $1024 \times 1024$ pixels represent $32° \times 32°$ of the visual field;
# if centered, the visual extent of this stimulus subtends
# from $-16°$ on the left, to $16°$ on the right,
# and from $-16°$ on top, to $16°$ on the bottom.

# %% Load example stimulus
stimulus = np.load("example_stimulus.npy")

# visual extent, in degrees visual angle,
# same convention as pyplot (left, right, top, bottom):
#
# NOTE that Robinson, Hammon, & de Sa (2007) actually implement
# a visual extent slightly smaller: (1023/32)
visextent = tuple(np.asarray((-0.5, 0.5, -0.5, 0.5)) * (1023 / 32))

# Visualise
plt.subplot(1, 2, 1)
plt.imshow(stimulus, cmap="gray", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(1, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    stimulus[512, 256:768],
    color="black",
)

plt.show()

# %% [markdown]
# In the stimulus image on the left, the left gray patch appears brighter than the right gray patch.
# On the right, the pixel intensity/gray scale values along the horizontal cut
# (indicated by the dashed line in the image) are shown.
# These reveal that, in fact, the two gray patches are identical in their physical intensity.

# %% [markdown]
# ## Image space
# To construct any filters, first the coordinates of the space must be defined.
# Here, the stimulus is said to subtend $32 \times 32$ degrees of visual angle,
# and has a resolution of $1024 \times 1024$ pixels.
# The horizontal (`axish`) and vertical (`axisv`) axes of the space
# sample the range set by the limits of the visual extent, in $1024$ steps.
# The `numpy.meshgrid()` function produces arrays with the `x` and `y` coordinate,
# in degrees of visual angle, of each of the $1024 \times 1024$ pixels in the image(space).
#
# Defining the coordinates in degrees of visual angle
# allows for defining all other space quantities also in degrees.
# If, instead, the coordinates are in pixels,
# all other quantities (e.g., size of filters)
# must also be given in pixels.

# %%
axish = np.linspace(visextent[0], visextent[1], stimulus.shape[0])
axisv = np.linspace(visextent[2], visextent[3], stimulus.shape[1])

(x, y) = np.meshgrid(axish, axisv)

# %% [markdown]
# ## Filter types
# The `multyscale.filters` module implements several filters:
#
# - `multyscale.filters.gaussian2d`: a two-dimensional Gaussian filter,
#    which is defined by a standard deviation along a major and minor axis,
#    and optionally a rotation
# - `multyscale.filters.dog`: a isotropic difference-of-Gaussian filter,
#    composed of a center 2D Gaussian and a surround 2D Gaussian,
#    which are each symmetrical
#    (i.e., standard deviation along major and minor axes are identical)
# - `multyscale.filters.odog`: an oriented difference-of-Gaussian filter,
#   composed of a center 2D Gaussian and  a surround 2D Gaussian,
#   which do not have to be isotropic
#   (i.e., can have different standard deviations along their major and minor axes).

# %% [markdown]
# ### Two-dimensional Gaussian
# The two-dimensional Gaussian is evaluate over the whole input space,
# i.e., for each coordinate in `x` and `y`.
# The shape of a Gaussian is generally defined by two parameters:
# its central tendency, and its spread.
# For a 2D Gaussian in space, this corresponds to the `center` location,
# and the standard deviation `sigma`.
# The present implementation of the Gaussian defaults to `center`ing on `(0,0)`.

# %% 2D Gaussians, both circular and elliptic
gaussian_circular = filters.gaussian2d(x, y, sigma=2)
gaussian_elliptic = filters.gaussian2d(x, y, sigma=1, center=(-4, 6))

# Plot
plt.subplot(2, 2, 1)
plt.imshow(gaussian_circular, extent=visextent, cmap="coolwarm")
plt.subplot(2, 2, 2)
plt.imshow(gaussian_elliptic, extent=visextent, cmap="coolwarm")

# Plot horizontal meridians
plt.subplot(2, 2, 3)
plt.plot(
    x[int(gaussian_circular.shape[0] / 2)],
    gaussian_circular[int(gaussian_circular.shape[0] / 2), ...],
)
plt.subplot(2, 2, 4)
plt.plot(
    x[int(gaussian_elliptic.shape[0] / 2)],
    gaussian_elliptic[int(gaussian_elliptic.shape[0] / 2), ...],
)

plt.show()

# %% [markdown]
# ### Oriented 2D Gaussian
# The previous filters are isotropic: they are radially symmetric and thus _unoriented_.
# This is because the 2D Gaussian(s) have the same standard deviation along both axes.
# However, this is not required.
# _Oriented_ 2D Gaussians can be made by differing the standard deviation along the two axes.
# Optionally, the major and minor axes of the Gaussian
# can be rotated away from the horizontal and vertical axes.

# %% Oriented Gaussians filters
gaussian_circular = filters.gaussian2d(x, y, sigma=(6, 2))
gaussian_elliptic = filters.gaussian2d(
    x, y, sigma=(6, 2), orientation=65
)  # rotation is counterclockwise

# % Visualize
fig, axs = plt.subplots(2, 2, sharex="all", sharey="row")

# Plot images
axs[0, 0].imshow(gaussian_circular, extent=visextent, cmap="coolwarm")
axs[0, 1].imshow(gaussian_elliptic, extent=visextent, cmap="coolwarm")

# Plot horizontal meridians
axs[1, 0].plot(
    x[int(gaussian_circular.shape[0] / 2)],
    gaussian_circular[int(gaussian_circular.shape[0] / 2), ...],
)
axs[1, 1].plot(
    x[int(gaussian_elliptic.shape[0] / 2)],
    gaussian_elliptic[int(gaussian_elliptic.shape[0] / 2), ...],
)

plt.show()

# %% [markdown]
# ### Difference-of-Gaussian
# A common type of filter of image processing, is the difference-of-Gaussian (DoG) filter.
# As the name implies, it consists of two, 2D Gaussian filters:
# a smaller "center" Gaussian, and a larger "surround" Gaussian,
# and the filter subtracts the surround Gaussian from the center Gaussian.
#
# For an isotropic DoG filter only one standard deviation is required
# for each of the constituent Gaussians.
# These filters are radially symmetric
# and therefore considered _unoriented_ DoG filters.

# %% Difference-of-Gaussian
filt_dog = filters.dog(x, y, sigma=(2, 4))  # surround Gaussian is 2:1 of center gaussian

# Plot
plt.subplot(1, 2, 1)
plt.imshow(filt_dog, extent=visextent, cmap="coolwarm")

# Plot horizontal meridian
plt.subplot(1, 2, 2)
plt.plot(x[int(filt_dog.shape[0] / 2)], filt_dog[int(filt_dog.shape[0] / 2), ...])

plt.show()

# %% [markdown]
# ### Oriented Difference-of-Gaussian
# A difference-of-Gaussian filter also does not have to be isotropic.
# If at least one of the constituent Gaussians (center or surround) is oriented,
# so will the combined filter be.
# This filter now is an _oriented_ difference-of-Gaussians (ODoG) filter.

# %% ODOG filter
sigma = ((2, 2), (2, 4))  # surround Gaussian is 2:1 in one axis
filt_odog = filters.odog(x, y, sigma, orientation=80)

# Plot filter and horizontal meridian
plt.subplot(1, 2, 1)
plt.imshow(filt_odog, extent=visextent, cmap="coolwarm")
plt.subplot(1, 2, 2)
plt.plot(x[int(filt_odog.shape[0] / 2)], filt_odog[int(filt_odog.shape[0] / 2), ...])

plt.show()


# %% [markdown]
# ## Applying filters
# The `multyscale.filters.apply()` function is used to apply
# a filter (as an `numpy.NDArray`) to an image (also an `numpy.NDArray`).
#
# If the filter and image are *not* the same `shape` (in pixels),
# the smaller will be _padded_ to match the larger one.
# The `padval` argument specifies what value will be used, by default `0.5`.

# %% ODOG filter
filt_odog = filters.odog(x, y, sigma=((1, 1), (1, 2)), orientation=90)

# % Apply filter
filtered_img = filters.apply(stimulus, filt_odog)

# Plot stimulus
plt.subplot(1, 3, 1)
plt.imshow(stimulus, cmap="gray", extent=visextent)

# Plot filter + horizontal meridian
plt.subplot(2, 3, 2)
plt.imshow(filt_odog, extent=visextent, cmap="coolwarm")
plt.subplot(2, 3, 5)
plt.plot(x[int(filt_odog.shape[0] / 2)], filt_odog[int(filt_odog.shape[0] / 2), ...])

# Plot filtered image
plt.subplot(1, 3, 3)
plt.imshow(filtered_img, cmap="coolwarm", extent=visextent)

plt.show()
