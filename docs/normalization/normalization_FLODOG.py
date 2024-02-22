# %% [markdown]
# # FLODOG normalization
# This Tutorial describes the rationale behind
# the normalization step of the FLODOG model (Robinson, Hammon, de Sa 2007),
# and its implementation in `multyscale`.

# %% Setup
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
# ## FLODOG: making -ODOG more biologically plausible
#
# Robinson, Hammon, & de Sa (2007) argue for two changes to the -ODOG models,
# based on known properties of the visual system.
#
# Firstly, the divisive normalization should be spatially local.
# This is partly implemented in the LODOG normalization step (see normalization_LODOG).
# However, it would be more plausible if the spatial window for normalization
# depends on scale of the filter being normalized:
# since small filters operate on small spatial areas,
# the normalization should similarly only be affected by nearby activity;
# and large scale filter should be affected by energy in more distal regions.
#
# Secondly, they argue that because similar spatial frequencies cluster in early visual regions,
# they are also more likely to inhibit each other.
# In other words, filters at similar scales (sensitive to similar spatial frequencies)
# should normalize each other more strongly than filters with very different sensitivies.
#
# Thus, Robinson, Hammon, & de Sa (2007) propose a
# (spatial) **F**requency-specific **L**ocally normalized -ODOG (FLODOG) model,
# which builds on the LODOG model but differs in these two aspects.

# %%
LODOG = multyscale.models.LODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
FLODOG = multyscale.models.FLODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
assert np.array_equal(FLODOG.bank.filters, LODOG.bank.filters)


# %% [markdown]
# The models share the same filterbank frontend,
# so it's only necessary to apply this bank
# (and weight the filteroutputs)
# once.

# %%
filters_output = FLODOG.bank.apply(stimulus)
filters_output = FLODOG.weight_outputs(filters_output)

# %% [markdown]
# ## FLODOG normalization
# Generally, the -ODOG normalization step consists of three parts:
#
# 1. The _normalizing coefficients_: weighted combinations of all filter outputs
# 2. Energy-calculating (as spatial averaging) of the normalizing coefficients
# 3. Divisive normalization, where a filter output is divided by the energy (2)
#    of its normalizing coefficient (1)
#
# The FLODOG differs from the (L)ODOG models in both normalizing coefficients (1),
# and how energy is estimated (2).

# %% [markdown]
# ### Normalizing coefficients
# Where the (L)ODOG models use equal weighting
# for filters of all scales to construct the normalizing coefficients $\mathbf{N}$,
# the FLODOG normalizing coefficients weight more heavily
# those spatial scales that are more similar to the filter being normalized
# than more different scales.

# %% [markdown]
# Instead of the equal weighting,
# the FLODOG model uses a 1D Gaussian as the weights profile:
# centered on the spatial scale of the filter being normalized
# and dropping off as a Gaussian function of the relative index the other spatial scales.

# %% Define weights
scale_norm_weights_FLODOG = multyscale.normalization.scale_norm_weights_gaussian(7, sdmix=0.5)
assert np.array_equal(scale_norm_weights_FLODOG, FLODOG.scale_norm_weights)

# %%
fig, axs = plt.subplots(2, 2, sharex="row", sharey="row")
axs[0, 0].pcolor(
    LODOG.scale_norm_weights,
    cmap="Greens",
    edgecolors="k",
    linewidths=1,
    vmin=0,
    vmax=1,
)
axs[0, 0].set_ylabel("scale of filter to normalize (idx)")
axs[0, 0].set_title("LODOG weights")

axs[0, 1].pcolor(
    FLODOG.scale_norm_weights,
    cmap="Greens",
    edgecolors="k",
    linewidths=1,
    vmin=0,
    vmax=1,
)
axs[0, 1].set_xlabel("scale of filter to normalize (idx)")
axs[0, 1].set_title("FLODOG weights")

axs[1, 0].plot(LODOG.scale_norm_weights[3, :], color="black")
axs[1, 0].set_xlabel("scale of other filter (idx)")
axs[1, 0].set_ylabel("weight")

axs[1, 1].plot(FLODOG.scale_norm_weights[3, :], color="black")
axs[1, 1].set_xlabel("scale of other filter (idx)")

plt.show()

# %% [markdown]
# Since these weights are strongly biased towards same/similar spatial scales,
# the resulting normalizing coefficients also more strongly resemble
# the filter outputs at these spatial scales.

# %% Normalizing coefficients
normalizing_coefficients_LODOG = LODOG.norm_coeffs(filters_output)
normalizing_coefficients_FLODOG = FLODOG.norm_coeffs(filters_output)

# Visualize each norm. coeff.
vmin = min(np.min(normalizing_coefficients_LODOG), np.min(normalizing_coefficients_FLODOG))
vmax = max(np.max(normalizing_coefficients_LODOG), np.max(normalizing_coefficients_FLODOG))

fig, axs = plt.subplots(*normalizing_coefficients_LODOG.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalizing_coefficients_LODOG.shape[:2]):
    axs[o, s].imshow(
        normalizing_coefficients_LODOG[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=vmin,
        vmax=vmax,
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
fig.suptitle("LODOG")
plt.show()

fig, axs = plt.subplots(*normalizing_coefficients_FLODOG.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalizing_coefficients_FLODOG.shape[:2]):
    axs[o, s].imshow(
        normalizing_coefficients_FLODOG[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=vmin,
        vmax=vmax,
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
fig.suptitle("FLODOG")
plt.show()


# %% [markdown]
# ### Scale-localized energy estimate
# Secondly, the LODOG and FLODOG models differ in
# the Gaussian spatial averaging window that they use
# to locally calculate the energy (spatial RMS) of each normalization coefficients.

# Where the LODOG model uses a single Gaussian window for all normalization coefficients,
# in FLODOG, the width of this spatial averaging window scales with the scale of the filter
# being normalized.
#
# Thus the normalization of the **F**LODOG model is "localized"
# both in space, and in _spatial scale_ / _**F**requency_.

# %% Spatial averaging window
spatial_windows_LODOG = np.ndarray(filters_output.shape)
for o, s in np.ndindex(LODOG.window_sigmas.shape[:2]):
    spatial_windows_LODOG[o, s, :] = multyscale.normalization.spatial_kernel_gaussian(
        LODOG.bank.x, LODOG.bank.y, LODOG.window_sigmas[o, s, :]
    )

spatial_windows_FLODOG = np.ndarray(filters_output.shape)
for o, s in np.ndindex(FLODOG.window_sigmas.shape[:2]):
    spatial_windows_FLODOG[o, s, :] = multyscale.normalization.spatial_kernel_gaussian(
        FLODOG.bank.x, FLODOG.bank.y, FLODOG.window_sigmas[o, s, :]
    )

# Visualize each spatial avg. window
fig, axs = plt.subplots(2, spatial_windows_LODOG.shape[1], sharex="all", sharey="all")
for s in range(spatial_windows_LODOG.shape[1]):
    axs[0, s].imshow(
        spatial_windows_LODOG[3, s],
        cmap="coolwarm",
        extent=visextent,
    )
    axs[1, s].imshow(
        spatial_windows_FLODOG[3, s],
        cmap="coolwarm",
        extent=visextent,
    )
axs[0, 3].set_title("LODOG")
axs[1, 3].set_title("FLODOG")
plt.show()

# %% [markdown]
# Both these factors influence the final energy estimates
# that form the denominators of the normalization

# %% Energy estimates
energies_LODOG = LODOG.normalizers_to_RMS(normalizing_coefficients_LODOG)
energies_FLODOG = FLODOG.normalizers_to_RMS(normalizing_coefficients_FLODOG)

# Visualize
vmin = min(np.min(energies_LODOG), np.min(energies_FLODOG))
vmax = max(np.max(energies_LODOG), np.max(energies_FLODOG))
fig, axs = plt.subplots(*energies_LODOG.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(energies_LODOG.shape[:2]):
    axs[o, s].imshow(
        energies_LODOG[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=vmin,
        vmax=vmax,
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
fig.suptitle("LODOG")
plt.show()

fig, axs = plt.subplots(*energies_FLODOG.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(energies_FLODOG.shape[:2]):
    axs[o, s].imshow(
        energies_FLODOG[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=vmin,
        vmax=vmax,
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
fig.suptitle("FLODOG")
plt.show()

# %% [markdown]
# With these parameters,
# the normalizing energy coefficients (i.e., denominators in normalization)
# of the FLODOG model are much more spatially localized,
# and look much more similar to the filters that they are normalizing.
# Moreover, in absolute terms, the values are a lot smaller,
# meaning stronger enhancement of channels

# %%
pd.DataFrame(
    {
        "LODOG": pd.Series({"min": np.min(energies_LODOG), "max": np.max(energies_LODOG)}),
        "FLODOG": pd.Series({"min": np.min(energies_FLODOG), "max": np.max(energies_FLODOG)}),
    }
)

# %% [markdown]
# As a result,
# the FLODOG normalization results in much stronger local "peaks"
# and thus more spatial inhomogeneities

# %% Normalize
LODOG_normalized = filters_output / energies_LODOG
FLODOG_normalized = filters_output / energies_FLODOG

# Visualize normalized outputs
vmin = min(np.min(LODOG_normalized), np.min(FLODOG_normalized))
vmax = max(np.max(LODOG_normalized), np.max(FLODOG_normalized))
fig, axs = plt.subplots(*LODOG_normalized.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(LODOG_normalized.shape[:2]):
    axs[o, s].imshow(
        LODOG_normalized[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=vmin,
        vmax=vmax,
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
fig.suptitle("LODOG")
plt.show()

fig, axs = plt.subplots(*FLODOG_normalized.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(FLODOG_normalized.shape[:2]):
    axs[o, s].imshow(
        FLODOG_normalized[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=vmin,
        vmax=vmax,
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
fig.suptitle("FLODOG")
plt.show()


# %% [markdown]
# Similarly, this then shows up in the final integrated outputs

# %% Integrated
output_LODOG = LODOG_normalized.sum((0, 1))
output_FLODOG = FLODOG_normalized.sum((0, 1))

# Visualize outputs
vmin = min(np.min(output_LODOG), np.min(output_FLODOG))
vmax = max(np.max(output_LODOG), np.max(output_FLODOG))

plt.subplot(2, 2, 1)
plt.imshow(output_LODOG, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=visextent)
plt.title("LODOG")
plt.subplot(2, 2, 2)
plt.imshow(output_FLODOG, cmap="coolwarm", vmin=vmin, vmax=vmax, extent=visextent)
plt.title("FLODOG")
plt.subplot(2, 1, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    output_LODOG[512, :],
    color="black",
    linestyle="solid",
    label="LODOG",
)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1]),
    output_FLODOG[512, :],
    color="black",
    linestyle="dotted",
    label="FLODOG",
)
plt.legend()
plt.show()

# %% [markdown]
# The "more biological plausible" FLODOG model predicts
# the direction of many brightness stimuli well,
# including ones that the (L)ODOG models fail to predict.
# However, it also produces these strong spatial inhomogeneities,
# which strongly emphasize borders and are clearly visible in the model outputs.
# These inhomogeneities are not immediately perceptually experienced in the original stimuli,
# though some suggest they can be measured psychophysically.
#
# From a modeling point of view, it is also important to recognize that the output ranges,
# i.e., the units on the vertical axes, are vastly different.
# Thus, for meaningful quantitative comparisons,
# some form of standardization is required.
