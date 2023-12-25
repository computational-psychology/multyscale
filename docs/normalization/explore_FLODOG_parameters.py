# %% [markdown]
# # Exploring FLODOG normalization parameter
# This guide describes how to change the normalization parameters
# of the FLODOG model
# and explores how this affects model output.
# This variation is also a replication of Robinson, Hammon, & de Sa (2007; fig 5).

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

# %% [markdown]
# In the stimulus image on the left, the left gray patch appears brighter than the right gray patch.
# On the right, the pixel intensity/gray scale values along the horizontal cut
# (indicated by the dashed line in the image) are shown.
# These reveal that, in fact, the two gray patches are identical in their physical intensity.

# %% [markdown]
# ## Exploring FLODOG normalization
# The FLODOG model differs from the (L)ODOG models in its normalization step
# (see normalization_FLODOG).
# The parameters we are exploring here, only affect this normalization step.
# Thus, we can apply differently parameterized normalizations
# to the same (weighted) filterouputs.

# %% Instantiate model
FLODOG = multyscale.models.FLODOG_RHS2007(shape=stimulus.shape, visextent=visextent)

# %% Apply filterbank to (example) stimulus
filters_output = FLODOG.bank.apply(stimulus)
filters_output = FLODOG.weight_outputs(filters_output)

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

# %% Output default model
normalized_4_05 = FLODOG.normalize_outputs(filters_output)
output_4_05 = np.sum(normalized_4_05, axis=(0, 1))

# %% [markdown]
# Generally, the -ODOG normalization steps consist of three parts:
#
# 1. The _normalizing coefficients_: weighted combinations of all filter outputs
# 2. Energy-calculating (as spatial averaging) of the normalizing coefficients
# 3. Divisive normalization, where a filter output is divided by the energy (2)
#    of its normalizing coefficient (1)
#
# Which (other) filter outputs get weighted stronger (1) in the normalizing coefficients,
# differs between the models.
# The (L)ODOG models weight every filer with the same orientation, equally.
# The FLODOG model biases the normalization towards similar spatial scales:
# the weight is a (1D) Gaussian function of the difference between
# the scale of the filter being normalized, and the other scales.
# It's centered around 0, such that filters are most strongly normalized by themselves.
# The width of this Gaussian weighting function is specified using
# the parameter `sdmix`:
# $\sigma_w = \mathrm{sdmix} \times s$

# %% FLODOG, with sdmix=3
# Set parameter
sdmix = 3
# FLODOG.sdmix = sdmix

# Determine weight
scale_norm_weights = multyscale.normalization.scale_norm_weights_gaussian(
    len(FLODOG.scale_weights), sdmix
)
# FLODOG.scale_norm_weights = scale_norm_weights
normalization_weights = multyscale.normalization.create_normalization_weights(
    n_orientations=FLODOG.bank.shape[0],
    n_scales=FLODOG.bank.shape[1],
    scale_norm_weights=scale_norm_weights,
    orientation_norm_weights=FLODOG.orientation_norm_weights,
)
# FLODOG.normalization_weights = normalization_weights

# %% Determine normalizing coefficients
normalizing_coefficients_3 = multyscale.normalization.normalizers(
    filters_output, normalization_weights
)

# Visualise each normalizing coefficient
fig, axs = plt.subplots(*normalizing_coefficients_3.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalizing_coefficients_3.shape[:2]):
    axs[o, s].imshow(
        normalizing_coefficients_3[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=normalizing_coefficients_3.min(),
        vmax=normalizing_coefficients_3.max(),
    )
fig.supxlabel("Spatial scale/freq. $s$")
fig.supylabel("Orientation $o$")
plt.show()

# %% [markdown]
# The (F)LODOG models also perform a local spatial averaging (2)
# to calculate the energy of the normalizing coefficients.
# The size of the local averaging window will affect the model predictions.
# The FLODOG model specifically scales the local averaging window
# (the $\sigma$ of the 2D Gaussian)
# to the spatial scale of the filter being normalized.
# This is controlled by a parameter `spatial_window_scalar`:
# $\sigma = \mathrm{spatial_window_scalar} \times s$.
# To explore how, here we apply various parameterizations of this normalization
# to the same stimulus image and corresponding (weighted) filter outputs.

# %% FLODOG with spatial_window_scalar = 2
# Set parameter
spatial_window_scalar = 2
# FLODOG.spatial_window_scalar = spaial_window_scalar

# Create spatial window sigmas
window_sigmas = spatial_window_scalar * np.broadcast_to(
    np.array(FLODOG.center_sigmas)[None, ..., None], (*FLODOG.bank.shape[:2], 2)
)
FLODOG.window_sigmas = window_sigmas

# Apply spatial averaging windows to normalizing coefficients
energies_2_3 = FLODOG.normalizers_to_RMS(normalizing_coefficients_3)

# Visualize each energy estimate
fig, axs = plt.subplots(*energies_2_3.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(energies_2_3.shape[:2]):
    axs[o, s].imshow(
        energies_2_3[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=energies_2_3.min(),
        vmax=energies_2_3.max(),
    )
fig.supxlabel("Spatial scale/freq. $s$")
fig.supylabel("Orientation $o$")
plt.show()

# %% Output
normalized_2_3 = filters_output / energies_2_3
output_2_3 = np.sum(normalized_2_3, axis=(0, 1))

# %% Comparing FLODOG outputs
vmin = min(np.min(output_4_05), np.min(output_2_3))
vmax = max(np.max(output_4_05), np.max(output_2_3))

plt.subplot(2, 2, 1)
plt.imshow(output_4_05, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=visextent)
plt.subplot(2, 2, 2)
plt.imshow(output_2_3, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=visextent)
plt.subplot(2, 1, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    output_4_05[512, :],
    color="black",
    linestyle="dotted",
    label="FLODOG $\sigma=4; sdmix=0.5$",
)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    output_2_3[512, :],
    color="black",
    linestyle="dashed",
    label="FLODOG $\sigma=2s; sdmix=1$",
)
plt.legend()
plt.show()


# %% [markdown]
# ### Construct different models
# An easy(r) way to explore different values for these parameter,
# is to create a different instance of the `FLODOG_RHS2007` model-class
# passing in the `sdmix` and/or spatial_window_scaler parameter(s) to the constructor.

# %% Instantiate new model object
FLODOG_4_3 = multyscale.models.FLODOG_RHS2007(
    shape=stimulus.shape, visextent=visextent, spatial_window_scalar=4, sdmix=3
)

# %% [markdown]
# Since the model-object has then been constructed already with the different attributes,
# we can just run the `.normalize_outputs()` method
# to normalize the filter outputs accordingly.

# %% Normalize
normalized_4_3 = FLODOG_4_3.normalize_outputs(filters_output)

# %% [markdown]
# To then readout the final model prediction,
# we sum over orientations and spatial scales.

# %% Readout
output_4_3 = np.sum(normalized_4_3, axis=(0, 1))

# Visualize
vmin = min(np.min(output_4_05), np.min(output_4_3))
vmax = max(np.max(output_4_05), np.max(output_4_3))

plt.subplot(2, 2, 1)
plt.imshow(output_4_05, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=visextent)
plt.subplot(2, 2, 2)
plt.imshow(output_4_3, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=visextent)
plt.subplot(2, 1, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    output_4_05[512, :],
    color="black",
    linestyle="dotted",
    label="FLODOG $\sigma=4; sdmix=0.5$",
)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    output_4_3[512, :],
    color="black",
    linestyle="dashed",
    label="FLODOG $\sigma=2s; sdmix=1$",
)
plt.legend()
plt.show()

# %% [markdown]
# ## Recreate Robinson, Hammon, & de Sa (2007, Fig. 5)
# Robinson, Hammon, & de Sa (2007) compare
# three sets of parameters,
# varying the window size scaling factor between $\sigma=2s$ and $\sigma=4s$,
# and varying the width of the Gaussian defining the mixing weights between $m=0.5$ and $m=3.0$.
#
# Figure 5 shows and compares the spatial inhomogeneities
# of the three different parameterizations of the model.
# Also, these three versions of FLODOG are compared to the default LODOG model ($\sigma=4$).

# %% FLODOG parameterizations
# FLODOG_4_05 = multyscale.models.FLODOG_RHS2007(
#     shape=stimulus.shape, visextent=visextent, spatial_window_scalar=4, sdmix=0.5
# )
FLODOG_2_05 = multyscale.models.FLODOG_RHS2007(
    shape=stimulus.shape, visextent=visextent, spatial_window_scalar=2, sdmix=0.5
)
# FLODOG_4_3 = multyscale.models.FLODOG_RHS2007(
#     shape=stimulus.shape, visextent=visextent, spatial_window_scalar=4, sdmix=3
# )

# %% Outputs
output_2_05 = FLODOG_2_05.normalize_outputs(filters_output).sum((0, 1))

# %% LODOG, $\sigma=4$
LODOG = multyscale.models.LODOG_RHS2007(shape=stimulus.shape, visextent=visextent, window_sigma=4)
output_LODOG_4 = LODOG.normalize_outputs(filters_output).sum((0, 1))


# %% [markdown]
# To more properly compare the outputs of these different model instances,
# Robinson, Hammon, & de Sa (2007) standardize model outputs such that
# the difference between the mean output in the two target regions
# on the `WE_thick` stimulus
# is equal to $1.0$.
#
# To implement this here, we:
# - load a mask image (as a `numpy.NDArray`) that defines the two target regions
# - apply this mask to model output to get model output in just the two target regions
# - calculate the mean for each target region
# - subtract the two means
# - divide the whole model output by this difference

# %% Standardize such that effect size = 1
mask = np.load("example_stimulus_mask.npy")


def target_diff(output, mask):
    left_target_mask = mask == 1
    right_target_mask = mask == 2

    left_target_output = output[left_target_mask]
    right_target_output = output[right_target_mask]
    return left_target_output.mean() - right_target_output.mean()


output_LODOG_4 /= target_diff(output_LODOG_4, mask)
output_4_05 /= target_diff(output_4_05, mask)
output_2_05 /= target_diff(output_2_05, mask)
output_4_3 /= target_diff(output_4_3, mask)

# %% [markdown]
# This, then, gives us the outputs scaled into the same range,
# which replicates Figure 5 from Robinson, Hammon, & de Sa (2007).

# %%
plt.figure(figsize=(6, 10))

ax = plt.subplot(3, 1, 1)
ax.imshow(stimulus[275:750, :], cmap="gray")
ax.axhline(y=235, color="black", dashes=(1, 1))


plt.subplot(6, 1, 3)
plt.plot(output_LODOG_4[512, :], color="blue", linestyle="solid", label="LODOG n=4")
plt.grid(axis="y")
plt.yticks(ticks=range(-4, 6, 2))
plt.ylim(-4, 4)
plt.xlim(0, output_LODOG_4.shape[0])
plt.legend()


plt.subplot(6, 1, 4)
plt.plot(
    output_4_05[512, :],
    color="blue",
    linestyle="solid",
    label="FLODOG $\sigma=4s, m=0.5$",
)
plt.grid(axis="y")
plt.yticks(ticks=range(-4, 6, 2))
plt.ylim(-4, 4)
plt.xlim(0, output_4_05.shape[0])
plt.legend()


plt.subplot(6, 1, 5)
plt.plot(
    output_2_05[512, :],
    color="blue",
    linestyle="solid",
    label="FLODOG $\sigma=2s, m=0.5$",
)
plt.grid(axis="y")
plt.yticks(ticks=range(-4, 6, 2))
plt.ylim(-4, 4)
plt.xlim(0, output_2_05.shape[0])
plt.legend()


plt.subplot(6, 1, 6)
plt.plot(
    output_4_3[512, :],
    color="blue",
    linestyle="solid",
    label="FLODOG $\sigma=4s, m=3.0$",
)
plt.grid(axis="y")
plt.yticks(ticks=range(-4, 6, 2))
plt.ylim(-4, 4)
plt.xlim(0, output_4_3.shape[0])
plt.legend()


plt.show()
