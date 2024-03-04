# %% [markdown]
# # Bank of filters

# %% Setup
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np

# Import local module
import multyscale

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

# %% Image coordinate system:
axish = np.linspace(visextent[0], visextent[1], stimulus.shape[0])
axisv = np.linspace(visextent[2], visextent[3], stimulus.shape[1])

(x, y) = np.meshgrid(axish, axisv)

# %% [markdown]
# In the stimulus image on the left, the left gray patch appears brighter than the right gray patch.
# On the right, the pixel intensity/gray scale values along the horizontal cut
# (indicated by the dashed line in the image) are shown.
# These reveal that, in fact, the two gray patches are identical in their physical intensity.

# %% [markdown]
# ## Difference-of-Gaussian bank
# The -DOG family of models starts with a _multiscale spatial filtering_ frontend.
# This consists of a set of filters, $\mathbf{F}$
# which span a range of spatial scales $S$

# %% Spacing of filters
# Scales (spatial frequency sensitivities)
num_scales = 7
largest_center_sigma = 3  # in degrees
center_sigmas = multyscale.utils.octave_intervals(num_scales) * largest_center_sigma
cs_ratio = 2  # center-surround ratio
sigmas = [((s, cs_ratio * s)) for s in center_sigmas]

# %% Create filterbank
bank_DOG = multyscale.filterbanks.DOGBank(sigmas, x, y)

# Visualise filterbank
fig, axs = plt.subplots(1, *bank_DOG.shape[:-2], sharex="all", sharey="all")
for s in np.ndindex(*bank_DOG.shape[:-2]):
    axs[s].imshow(bank_DOG.filters[s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s$")
plt.show()

# %% [markdown]
# In this visualisation of all filters,
# the columns differ in the spatial scale of the filter.

# %% [markdown]
# These filters are then convolved with the stimulus image.
#
# Filterbank-objects have an `apply(...)` method,
# which filters the input stimulus with the whole bank.
# The output is an $O \times S \times Y \times X$ tensor
# of channel responses.

# %% Apply filterbank to (example) stimulus
filters_output = bank_DOG.apply(stimulus)

# Visualise each filter output
fig, axs = plt.subplots(1, *filters_output.shape[:-2], sharex="all", sharey="all")
for s in np.ndindex(filters_output.shape[:-2]):
    axs[s].imshow(
        filters_output[s],
        cmap="coolwarm",
        extent=visextent,
        vmin=filters_output.min(),
        vmax=filters_output.max(),
    )
fig.supxlabel("Spatial scale/freq. $s$")
plt.show()


# %% [markdown]
# ## ODoG bank
# The _oriented_ -ODOG subfamily of models uses filters
# that also have one of several orientations $O$.
# Thus, filter $f_{o,s}$ is a single filter in the set $\mathbf{F}$
# with orientation $o$ and scale $s$.
# Since each filter is 2D, it also has an implied $x,y$ pixels.
#
# As a result, we can also think of the 2D ($O\times S$) set $\mathbf{F}$ of filter(outputs),
# where each filter(output) $f_{o,s}$ is an image,
# as a 4D ($O \times S \times X \times Y$) set $\mathbf{I}$ of pixel intensities.
# $$ \mathbf{I}_{O \times S \times X \times Y} \equiv \mathbf{F}_{O \times S} $$
#
# For the current topic, we use the default ODOG filterbank,
# which can be created by `multyscale.filterbanks.RHS2007()`,
# with 6 orientations, and 7 spatial scales.

# %% Number of filters
n_orientations = 6
num_scales = 7

# %% Spacing of filters

# Orientations
orientations = tuple(np.arange(0, 180, 180 / n_orientations))

# Scales (spatial frequency sensitivities)
largest_center_sigma = 3  # in degrees
center_sigmas = multyscale.utils.octave_intervals(num_scales) * largest_center_sigma
cs_ratio = 2  # center-surround ratio
sigmas = [((s, s), (s, cs_ratio * s)) for s in center_sigmas]

# %% Create filterbank
bank_ODOG = multyscale.filterbanks.ODOGBank(orientations, sigmas, x, y)

# Visualise filterbank
fig, axs = plt.subplots(*bank_ODOG.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(*bank_ODOG.shape[:2]):
    axs[o, s].imshow(bank_ODOG.filters[o, s, ...], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s$")
fig.supylabel("Orientation $o$")
plt.show()

# %% [markdown]
# In this visualisation of all filters,
# the rows differ in orientation of the filter
# and columns differ in the spatial scale of the filter.

# %% [markdown]
# These filters are then convolved with the stimulus image.
#
# Filterbank-objects have an `apply(...)` method,
# which filters the input stimulus with the whole bank.
# The output is an $O \times S \times Y \times X$ tensor
# of channel responses.

# %% Apply filterbank to (example) stimulus
filters_output = bank_ODOG.apply(stimulus)

# Visualise each filter output
fig, axs = plt.subplots(*filters_output.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(filters_output.shape[:2]):
    axs[o, s].imshow(
        filters_output[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=filters_output.min(),
        vmax=filters_output.max(),
    )
fig.supxlabel("Spatial scale/freq. $s$")
fig.supylabel("Orientation $o$")
plt.show()
