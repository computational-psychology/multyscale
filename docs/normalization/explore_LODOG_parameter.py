# %% [markdown]
# # Exploring LODOG normalization parameter
# This guide describes how to change the normalization parameter
# of the LODOG model
# and explores how this affects model output.
# This variation is also a replication of Robinson, Hammon, & de Sa (2007; fig 3).

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
# ## Exploring LODOG normalization
# The LODOG model differs from the original/base ODOG model in its normalization step
# (see normalization_LODOG).
# Specifically, where the ODOG normalization calculates the global energy
# as the spatial RMS image-wide,
# the LODOG normalization calculates _local_ RMS,
# by averaging within a Gaussian window.

# %% Instantiate model
LODOG = multyscale.models.LODOG_RHS2007(shape=stimulus.shape, visextent=visextent)

# %% Apply filterbank to (example) stimulus
filters_output = LODOG.bank.apply(stimulus)
filters_output = LODOG.weight_outputs(filters_output)

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


# %% [markdown]
# Generally, the -ODOG normalization steps consist of three parts:
#
# 1. The _normalizing coefficients_: weighted combinations of all filter outputs
# 2. Energy-calculating (as spatial averaging) of the normalizing coefficients
# 3. Divisive normalization, where a filter output is divided by the energy (2)
#    of its normalizing coefficient (1)
#
# The LODOG model performs a local, rather than image-wide, spatial averaging (2)
# to calculate the energy of the normalizing coefficients.
#
# The size of the local averaging window will affect the model predictions.
# To explore how, here we apply various parameterizations of this normalization
# to the same stimulus image and corresponding (weighted) filter outputs.

# %% [markdown]
# The normalizing coefficients (1) are not affected by
# the LODOG-specific spatial window parameter,
# and thus also only need to be calculated once for all the parameterizations explored here.

# %%
normalizing_coefficients = LODOG.norm_coeffs(filters_output)

# %% Gaussian spatial averaging window [markdown]
# The spatial averaging window in the LODOG model is
# a circular 2D Gaussian.
# Thus, its size is specified as a single standard deviation $\sigma$
# that defines the width of the Gaussian in both directions.
# The default is $\sigma = 4°$.
# The same $\sigma$ is used for all normalizing coefficients
# (though `multyscale` implements this not as a single Gaussian filter,
# but as a set of identical filters).

# %%
window_sigma = LODOG.window_sigma
print(f"sigma = {window_sigma} deg")
window_sigmas = LODOG.window_sigmas

assert np.all(window_sigmas == window_sigma)

spatial_filters = np.ndarray(filters_output.shape)
for o, s in np.ndindex(LODOG.window_sigmas.shape[:2]):
    spatial_filters[o, s, :] = multyscale.normalization.spatial_kernel_gaussian(
        LODOG.bank.x, LODOG.bank.y, LODOG.window_sigmas[o, s, :]
    )
plt.subplot(1, 2, 1)
plt.imshow(spatial_filters[0, 0, ...], cmap="coolwarm", extent=visextent)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    spatial_filters[0, 0, 512, :],
    color="black",
)
plt.axvline(x=-window_sigma, color="grey", linestyle="dashed")
plt.axvline(x=window_sigma, color="grey", linestyle="dashed")
plt.xlabel("X (deg. vis. angle)")
plt.ylabel("Y (deg. vis. angle)")
plt.show()

# %% [markdown]
# The model method `.normalizers_to_RMS()` automatically generates and applies
# these spatial averaging windows to the proved normalizing coeffiencts.
# This produces an $O \times S$ set of locally ($Y \times X$) calculated energies,
# one for each normalizaing coefficient.

# %%
energies_4 = LODOG.normalizers_to_RMS(normalizing_coefficients)

# Visualize each local energy
fig, axs = plt.subplots(*energies_4.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(energies_4.shape[:2]):
    axs[o, s].imshow(energies_4[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# These locally calculated energies form the denominator of each divisive normalization.
# Since the local energy tensor is the same $(O, S, X, Y)$ shape as the filter outputs
# we can simply divide.

# %%
normalized_outputs_4 = filters_output / energies_4

# Visualize each normalized output
fig, axs = plt.subplots(*normalized_outputs_4.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalized_outputs_4.shape[:2]):
    axs[o, s].imshow(normalized_outputs_4[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# Combining the normalized outputs gives a final model prediction.

# %%
output_LODOG_4 = np.sum(normalized_outputs_4, axis=(0, 1))

# %% [markdown]
# Adjusting spatial averaging window size
# The easiest way to adjust the window sigma,
# is to set the `window_sigma` attribute of the model-object.

# %%
LODOG.window_sigma = 1

# %% [markdown]
# Since `multyscale` actually implements the spatial averaging window
# separately for each normalizing coefficient (i.e., for each model-filter)
# this requires then to also set the whole set of ($O \times S$) `window_sigmas`:

# %%
LODOG.window_sigmas = np.ones(shape=(*LODOG.bank.shape[:2], 2)) * LODOG.window_sigma
assert np.all(LODOG.window_sigmas == LODOG.window_sigma)

# %%
spatial_filters = np.ndarray(filters_output.shape)
for o, s in np.ndindex(LODOG.window_sigmas.shape[:2]):
    spatial_filters[o, s, :] = multyscale.normalization.spatial_kernel_gaussian(
        LODOG.bank.x, LODOG.bank.y, LODOG.window_sigmas[o, s, :]
    )

# %% 
plt.subplot(1, 2, 1) 
plt.imshow(spatial_filters[0, 0, ...], cmap="coolwarm", extent=visextent)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    spatial_filters[0, 0, 512, :],
    color="black",
)
plt.axvline(x=-window_sigma, color="grey", linestyle="dashed")
plt.axvline(x=window_sigma, color="grey", linestyle="dashed")
plt.xlabel("X (deg. vis. angle)")
plt.ylabel("Y (deg. vis. angle)")
plt.show()

# %% [markdown]
# From here, the exact same steps can be taken to calculate these
# more locally restricted (smaller spatial averaging window) energies

# %%
energies_1 = LODOG.normalizers_to_RMS(normalizing_coefficients)

# Visualize each local energy
fig, axs = plt.subplots(*energies_4.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(energies_1.shape[:2]):
    axs[o, s].imshow(energies_1[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# What can be seen here, is that the more local energy calculation
# means much more spatial variation over the whole image,
# as would be expected.

# %%
normalized_outputs_1 = filters_output / energies_1

# Visualize each normalized output
vmin = min(np.min(normalized_outputs_4), np.min(normalized_outputs_1))
vmax = max(np.max(normalized_outputs_4), np.max(normalized_outputs_1))
fig, axs = plt.subplots(*normalized_outputs_1.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalized_outputs_1.shape[:2]):
    axs[o, s].imshow(normalized_outputs_1[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# Resultingly, the normalized filter outputs can also vary much more locally.

# %% [markdown]
# Combining the normalized outputs gives a final model prediction.

# %%
output_LODOG_1 = np.sum(normalized_outputs_1, axis=(0, 1))

# %% Comparing LODOG outputs
vmin = min(np.min(output_LODOG_4), np.min(output_LODOG_1))
vmax = max(np.max(output_LODOG_4), np.max(output_LODOG_1))

plt.subplot(2, 2, 1)
plt.imshow(output_LODOG_4, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=visextent)
plt.subplot(2, 2, 2)
plt.imshow(output_LODOG_1, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=visextent)
plt.subplot(2, 1, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    output_LODOG_4[512, :],
    color="black",
    linestyle="dotted",
    label="LODOG $\sigma=4$",
)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    output_LODOG_1[512, :],
    color="black",
    linestyle="dashed",
    label="LODOG $\sigma=1$",
)
plt.legend()
plt.show()

# %% [markdown]
# Here it can be seen that in the complete model output
# for the LODOG model with spatial filtering $\sigma = 1°$
# shows a lot more spatial inhomogeneity;
# the neutral gray background/surround shows strong inhomogeneity in the model output,
# but also the bars of carrier grating show stronger edge artifacts.

# %% [markdown]
# ### Construct different models
# An easier way to explore different values for this parameter,
# is to create a different instance of the `LODOG_RHS2007` model-class
# passing in the `window_sigma` parameter to the constructor.
#
# Since the model-object has then been constructed already with the different `window_sigma`,
# we can just run the `.normalize_outputs()` method
# to normalize the filter outputs accordingly.
#
# To then readout the final model prediction,
# we sum over orientations and spatial scales.

# %%
LODOG_2 = multyscale.models.LODOG_RHS2007(
    shape=stimulus.shape, visextent=visextent, window_sigma=2
)
output_LODOG_2 = LODOG_2.normalize_outputs(filters_output).sum((0, 1))

# %% [markdown]
# ## Recreate Robinson, Hammon, & de Sa (2007, Fig. 3)
# Robinson, Hammon, & de Sa (2007) compare
# three values for the $\sigma$ parameter
# controlling the size of the (local) spatial normalization window:
# $1°$, $2°$, and $4°$.
# Figure 3 shows and compares the spatial inhomogeneities
# of the three different parameterizations of the model.
# Also, these three versions of LODOG will be compared to the default ODOG model.

# %%
ODOG = multyscale.models.ODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
output_ODOG = ODOG.normalize_outputs(filters_output).sum((0, 1))


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


output_ODOG /= target_diff(output_ODOG, mask)
output_LODOG_1 /= target_diff(output_LODOG_1, mask)
output_LODOG_2 /= target_diff(output_LODOG_2, mask)
output_LODOG_4 /= target_diff(output_LODOG_4, mask)

# %% [markdown]
# This, then, gives us the outputs scaled into the same range,
# which perfectly replicates Figure 3 from Robinson, Hammon, & de Sa (2007).

# %%
plt.figure(figsize=(6, 10))

ax = plt.subplot(2, 1, 1)
ax.imshow(stimulus[275:750, :], cmap="gray")
ax.axhline(y=235, color="black", dashes=(1, 1))


plt.subplot(2, 1, 2)
plt.plot(output_ODOG[512, :], color="black", linestyle="solid", label="ODOG")
plt.plot(output_LODOG_4[512, :], color="grey", linestyle="dotted", label="LODOG N=4")
plt.plot(output_LODOG_2[512, :], color="grey", linestyle="solid", label="LODOG N=2")
plt.plot(output_LODOG_1[512, :], color="black", linestyle="dashed", label="LODOG N=1")

plt.grid(axis="y")
plt.yticks(ticks=range(-9, 12, 3))

plt.legend()

plt.show()
