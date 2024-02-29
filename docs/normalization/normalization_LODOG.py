# %% [markdown]
# # LODOG normalization
# This Tutorial describes the rationale behind
# the normalization step of the LODOG model (Robinson, Hammon, de Sa 2007),
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
# is an image containing two version of White's (1979) classic illusion,
# as also used by Robinson, Hammon, & de Sa (2007) as `WE_dual`.
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
stimulus = np.load("WE_dual.npy")

# visual extent, in degrees visual angle,
# same convention as pyplot (left, right, top, bottom):
#
# NOTE that Robinson, Hammon, & de Sa (2007) actually implement
# a visual extent slightly smaller: (1023/32)
visextent = tuple(np.asarray((-0.5, 0.5, -0.5, 0.5)) * (1023 / 32))

# Visualise
plt.subplot(2, 2, 1)
plt.imshow(stimulus, cmap="gray", extent=visextent)
plt.axhline(y=0, xmin=0, xmax=0.5, color="black", dashes=(1, 1))
plt.axvline(x=8, ymin=0.25, ymax=0.75, color="black", dashes=(1, 1))
plt.xlabel("x (deg. vis. angle)")
plt.ylabel("y (deg. vis. angle)")

plt.subplot(2, 2, 3)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[0:512],
    stimulus[512, 0:512],
    color="black",
    dashes=(1, 1),
)
plt.xlabel("x")
plt.ylabel("luminance (normalized)")

plt.subplot(2, 2, 2)
plt.plot(
    stimulus[256:768, 768],
    np.linspace(visextent[0], visextent[1], stimulus.shape[0])[256:768],
    color="black",
    dashes=(1, 1),
)
plt.ylabel("y")
plt.xlabel("luminance (normalized)")
plt.show()

# %% [markdown]
# In this image there are two White's stimuli:
# one oriented horizontally (left), and one oriented vertically (right).
# In both stimuli,
# the gray target patch embedded in the black phase of the grating
# looks brighter than the identical gray target embedded in the white phase.
# For the left stimulus, this means the left target looks brighter
# than the isoluminant right target
# (cut-through shown in bottom left panel).
# For the right stimulus, this means the top target looks brighter
# than the isoluminant bottom target
# (cut-through shown in top right panel).

# %% [markdown]
# ## ODOG model fails
#
# This image, with two stimuli in different regions,
# challenges the original ODOG model by Blakeslee & McCourt (1997)

# %% ODOG
ODOG = multyscale.models.ODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
output_ODOG = ODOG.apply(stimulus)

# %% Extract target predictions
targets_ODOG = []
mask = np.load("WE_dual_mask.npy")
for idx in np.unique(mask.astype(int)):
    if idx > 0:
        targets_ODOG.append(np.median(output_ODOG[mask == idx]))

# %% Visualise
plt.subplot(2, 2, 1)
plt.imshow(output_ODOG, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, xmin=0, xmax=0.5, color="black", dashes=(1, 1))
plt.axvline(x=8, ymin=0.25, ymax=0.75, color="black", dashes=(1, 1))
plt.xlabel("x (deg. vis. angle)")
plt.ylabel("y (deg. vis. angle)")

plt.subplot(2, 2, 3)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[0:512],
    output_ODOG[512, 0:512],
    color="black",
)
plt.xlabel("x")
plt.ylabel("brightness (ODOG)")

plt.subplot(2, 2, 2)
plt.plot(
    output_ODOG[256:768, 768],
    np.linspace(visextent[0], visextent[1], stimulus.shape[0])[256:768],
    color="black",
)
plt.ylabel("y")
plt.xlabel("Brightness (ODOG)")

plt.subplot(2, 2, 4)
plt.bar(x=["left", "right", "top", "bottom"], height=targets_ODOG, color=plt.colormaps["tab10"](0))
plt.ylim(-2, 2)
plt.axhline(y=0, linestyle="dashed", color="k")
plt.xlabel("Target region")
plt.ylabel("Brightness (ODOG; median)")

plt.show()

# %% [markdown]
# Here we see that the ODOG model incorrectly predicts that
# the targets in the black phases
# (left target in left/horizontal stimulus;
# top target in right/vertical stimulus)
# are _darker_ than the targets in the white phases.


# %% [markdown]
# **ODOG is affected by global energy**
# This unsuccessful prediction arises from the fact that
# the ODOG model normalizes each filter output at each location
# by the _image-wide_/_global_ energy of other filter outputs.
# This means that the (energy of the) right stimulus
# affects the normalization of the left stimulus,
# and vice versa.


# %% [markdown]
# ## LODOG model
# To overcome these kinds of issues, Robinson, Hammon, and de Sa (2007)
# developed the LODOG model variant.
# The LODOG model differs from the original/base ODOG model in its normalization step.
# Specifically, where the ODOG normalization calculates the global energy
# as the spatial RMS image-wide,
# the LODOG normalization calculates _local_ RMS,
# by averaging within a Gaussian window.
#
# Since the models differ only in the _normalization_ step,
# and not in the filterbank used in the _encoding_ step,
# the same filter(outputs) can be used for both models.

# %% Instantiate LODOG
LODOG = multyscale.models.LODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
assert np.array_equal(ODOG.bank.filters, LODOG.bank.filters)

# %% Run LODOG normalization on -ODOG filter output
filters_output = ODOG.bank.apply(stimulus)
weighted_outputs = ODOG.weight_outputs(filters_output)

norm_outputs = LODOG.normalize_outputs(weighted_outputs, eps=1e-6)
output_LODOG = norm_outputs.sum(axis=(0, 1))

# %% Extract target prediction
targets_LODOG = []
for idx in np.unique(mask.astype(int)):
    if idx > 0:
        targets_LODOG.append(np.median(output_LODOG[mask == idx]))

targets = pd.DataFrame(
    {
        "ODOG": targets_ODOG,
        "LODOG": targets_LODOG,
    },
    index=["Left", "Right", "Top", "Bottom"],
)

# %% Visualise
plt.subplot(2, 2, 1)
plt.imshow(output_LODOG, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, xmin=0, xmax=0.5, color="black", dashes=(1, 1))
plt.axvline(x=8, ymin=0.25, ymax=0.75, color="black", dashes=(1, 1))
plt.xlabel("x (deg. vis. angle)")
plt.ylabel("y (deg. vis. angle)")

plt.subplot(2, 2, 3)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[0:512],
    output_LODOG[512, 0:512],
    color="black",
)
plt.xlabel("x")
plt.ylabel("brightness (LODOG)")

plt.subplot(2, 2, 2)
plt.plot(
    output_LODOG[256:768, 768],
    np.linspace(visextent[0], visextent[1], stimulus.shape[0])[256:768],
    color="black",
)
plt.ylabel("y")
plt.xlabel("brightness (LODOG)")

ax = plt.subplot(2, 2, 4)
targets.plot(ax=ax, kind="bar")
plt.axhline(y=0.0, linestyle="dashed", color="k")
plt.ylim(-2, 2)
plt.xlabel("Target region")
plt.ylabel("Brightness (median)")

plt.show()


# %% [markdown]
# This LODOG model _can_ successfully predict that
# the targets in the black phases
# (left target in left/horizontal stimulus;
# top target in right/vertical stimulus)
# are _brighter_ than the targets in the white phases.

# %% [markdown]
# ## LODOG normalization (compared to ODOG)
# The power of the -ODOG models comes from their normalization step,
# in which information from all the filters regulates the activity of each filter output.
# Generally, this normalization consists of three parts:
#
# 1. The _normalizing coefficients_: weighted combinations of all filter outputs
# 2. Energy-calculating (as spatial averaging) of the normalizing coefficients
# 3. Divisive normalization, where a filter output is divided by the energy (2)
#    of its normalizing coefficient (1)

# %% [markdown]
# ### Normalizing coefficients
# The LODOG model uses the same _normalizing coefficients_ $\mathbf{N}$
# as the original ODOG model.
# Briefly, $\mathbf{N}$ consists of $6 \times 7$ 2D ($X\times Y$) matrices,
# (one _normalizing coefficient_ $n_{o',s'}$ per filter $f_{o',s'}$ to normalize).
# Each normalizing coefficient $n_{o', s'}$ is a weighted combination of all filter outputs $F$,
# where the weights $w_{o', s', o, s}$ can vary.

# %% Define weights
scale_norm_weights = multyscale.normalization.scale_norm_weights_equal(7)
orientation_norm_weights = multyscale.normalization.orientation_norm_weights(6)
norm_weights = multyscale.normalization.create_normalization_weights(
    *filters_output.shape[:2], scale_norm_weights, orientation_norm_weights
)
assert np.array_equal(norm_weights, LODOG.normalization_weights)

# %% Normalizing images as weighted combination (tensor dot-product) of filter outputs
normalizing_coefficients = multyscale.normalization.norm_coeffs(weighted_outputs, norm_weights)
assert np.array_equal(normalizing_coefficients, LODOG.norm_coeffs(weighted_outputs))

# Visualize each normalizing coefficient n_{o,s}, i.e. for each individual filter f_{o,s}
fig, axs = plt.subplots(*normalizing_coefficients.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalizing_coefficients.shape[:2]):
    axs[o, s].imshow(normalizing_coefficients[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")

# %% [markdown]
# NOTE that all normalizing coefficients within a row are identical,
# i.e., all filters of an orientation get the same normalizing coefficient in the ODOG model.

# %% [markdown]
# ### Normalize by energy
# Rather than normalizing by this weighted sum of all filter outputs at each pixel location,
# instead the -ODOG models normalize by
# the _energy_ of the normalizing coefficient.
# Energy here is expressed as the (spatial) root-mean-square of the signal.
#
# 1. Square each ($X \times Y = 1024 \times 1024$) pixel of the combined normalizing coefficient
# 2. Spatial average (e.g., mean) over pixels ($X, Y$)
# 3. Square-root of this mean
#
# $$ RMS(n_{o',s'}) = \sqrt{\mathrm{avg}(n_{o',s',x,y})^2} $$
#
# %% [markdown]
# The base ODOG normalized calculates an image-wide mean:

# %% Global image RMS
normalization_RMSs = np.ndarray(weighted_outputs.shape[:2])
for o, s in np.ndindex(normalizing_coefficients.shape[:2]):
    normalization_RMSs[o, s] = np.sqrt((normalizing_coefficients[o, s] ** 2).mean())

# Visualise
plt.pcolor(normalization_RMSs, cmap="Greens", edgecolors="k", linewidths=1, vmin=0, vmax=1)
plt.ylabel("Orientation $o'$")
plt.xlabel("Spatial scale $s'$")
plt.show()

# %% [markdown]
# This heatmap represents the RMS value of the normalizing coefficient
# that each filter gets normalized by.
#
# As mentioned, this image-wide calculation
# means that the (energy of the) right stimulus
# affects the normalization of the left stimulus,
# and vice versa.

# %% [markdown]
# #### L is for Local
# The way LODOG and ODOG differ, is in how the two models
# spatially average the normalizing energy.
#
# Robinson, Hammon, & de Sa (2007) consider the ODOG kind of global image averaging
# not plausible in the human visual system.
# Considering each pixel in the 4D filter output $F_{o,s,x,y}$
# to represent the "activity" of one "neural channel/unit",
# a global image mean would imply that a neural channel is influenced
# not just by its close neighbors with similar/overlapping receptive fields,
# but also by units responding to distant visual regions.
#
# Instead in the LODOG model, units are influenced only by (spatially) nearby units.
# Instead of the global image mean,
# the LODOG model uses a Gaussian window to average over pixels,
# giving the _local_ (estimate of) energy:
#
# $$ RMS_\mathrm{local}(n_{o',s'},\sigma) = \sqrt{G(\sigma) * (n_{o',s',x,y})^2}$$

# %% Gaussian averaging window
sigma = LODOG.window_sigma
window = multyscale.filters.gaussian2d(LODOG.bank.x, LODOG.bank.y, (sigma, sigma))

# Normalize window to unit-sum (== spatial averaging filter)
window = window / window.sum()

# Visualize Gaussian spatial averaging window
plt.subplot(2, 2, 1)
plt.imshow(window, extent=visextent, cmap="coolwarm")
plt.subplot(2, 2, 2)
plt.plot(LODOG.bank.x[int(window.shape[0] / 2)], window[int(window.shape[0] / 2), ...])
plt.show()

# %% [markdown]
# The function `multyscale.normalization.spatial_kernel_gaussian()`
# generates a ( $O\times S$ set of) Gaussian filters $G$,
# where each Gaussian spatial averaging window
# $G_{o',s'}$ is used to locally average filter $f_{o',s'}$.
# In the LODOG model, all $G$ are identical.
#
# The parameter $\sigma$ controls the spatial size of each $G(\sigma)$ Gaussian filter;
# thus this function takes in $O \times S$ $\sigma_{o', s'}$.
# In the LODOG model, all $\sigma_{o', s'}$ are identical.

# %% All sigmas identical
print(LODOG.window_sigmas)

# %% Generate Gaussian filters
spatial_windows = np.ndarray(filters_output.shape)
for o, s in np.ndindex(LODOG.window_sigmas.shape[:2]):
    spatial_windows[o, s, :] = multyscale.normalization.spatial_kernel_gaussian(
        LODOG.bank.x, LODOG.bank.y, LODOG.window_sigmas[o, s, :]
    )

assert np.array_equal(spatial_windows[0, 0, :], window)
assert np.array_equal(spatial_windows, LODOG.spatial_kernels())

idx = (2, 3)

plt.imshow(spatial_windows[*idx], cmap="coolwarm", extent=visextent)
plt.show()

# %% [markdown]
# Applying this Gaussian window gives the _local_ (estimate of) of energy

# %% Local energy estimates
normalization_local_energies = np.ndarray(normalizing_coefficients.shape)
for o, s in np.ndindex(normalizing_coefficients.shape[:2]):
    coeff = normalizing_coefficients[o, s] ** 2
    energy = multyscale.filters.apply(spatial_windows[o, s], coeff, padval=0)
    energy = np.sqrt(energy + 1e-6)  # minor offset to avoid negatives/0's
    normalization_local_energies[o, s, :] = energy

assert np.allclose(
    normalization_local_energies, LODOG.norm_energies(normalizing_coefficients, eps=1e-6)
)

# Visualize each local RMS
fig, axs = plt.subplots(*normalization_local_energies.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalization_local_energies.shape[:2]):
    axs[o, s].imshow(normalization_local_energies[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# NOTE that the _local_ energy here for each filter has to be a full $X \times Y$ pixel array,
# whereas the global energy in the base ODOG model is a single scalar value for each filter.
# Also NOTE that the energy in each _row_ (i.e., across scales) is constant,
# only along columns (i.e., across orientations) does the energy vary.
#
# Here we can see that
# the normalizing coefficients constructed from more vertically-oriented filer(outputs)
# -- top and bottom rows --
# have more energy on the right hand side of the image.
# Conversely,
# the normalizing coefficients constructed from more horizontally-oriented filter(outputs)
# -- middle rows --
# have more energy on the left hand side of the image.


# %% [markdown]
# ### Divisive normalization
# These matrices, expressing the local energy (Gaussian RMS) of the normalizing coefficients,
# form the denominator of the divisive normalization.
#
# Since the local energy is different in the two halves,
# we can see that the filters sensitive to vertical contrast (top/bottom row)
# will get normalized more strongly on the right hand side of the image.
# Conversely, the filters sensitive to horizontal contrast (middle row(s))
# will get normalized more strongly on the left hand side of the image.
# Thus, the filter outputs to the two halves of the image, i.e., to the two stimuli,
# get normalized quite differently.
#
# We now divide each filter(output) $f_{o',s'}$
# by the spatial RMS of the normalizing coefficient $n_{o',s'}$:
# $$f'_{o',s'} = \frac{f_{o',s'}}{RMS(n_{o',s'})}$$

# %% Divisive normalization
# Since the local RMSs tensor is the same (O, S, X, Y) shape as the filter outputs
# we can simply divide
normalized_outputs = weighted_outputs / (normalization_local_energies + 1e-6)
assert np.allclose(normalized_outputs, norm_outputs)

# Visualize each normalized f'_{o',s'}
fig, axs = plt.subplots(*normalized_outputs.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalized_outputs.shape[:2]):
    axs[o, s].imshow(normalized_outputs[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()


# %% [markdown]
# Thus, the full normalization schema of LODOG can be formulated as:
# $$ F' = \frac{f_{o',s',x,y}}{\sqrt{G(\sigma) *
# (\sum_{o=1}^{O}\sum_{s=1}^{S} w_{o',s',o,s}f_{o,s,x,y})^2}}$$
# where $w_{o', s', o, s} =   \begin{cases}
#       1 & o = o'  \\
#       0 & else
#    \end{cases}
# $

# %% [markdown]
# ## Readout
# Finally, we readout a final model output
# by recombining the normalized filter outputs
# -- simply by summing them all up.

# %% Recombine
recombined_outputs = np.sum(normalized_outputs, axis=(0, 1))
assert np.allclose(recombined_outputs, output_LODOG)

plt.subplot(2, 2, 1)
plt.imshow(recombined_outputs, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, xmin=0, xmax=0.5, color="black", dashes=(1, 1))
plt.axvline(x=8, ymin=0.25, ymax=0.75, color="black", dashes=(1, 1))
plt.xlabel("x (deg. vis. angle)")
plt.ylabel("y (deg. vis. angle)")

plt.subplot(2, 2, 3)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[0:512],
    recombined_outputs[512, 0:512],
    color="black",
)
plt.xlabel("x")
plt.ylabel("brightness (ODOG)")

plt.subplot(2, 2, 2)
plt.plot(
    recombined_outputs[256:768, 768],
    np.linspace(visextent[0], visextent[1], stimulus.shape[0])[256:768],
    color="black",
)
plt.ylabel("y")
plt.xlabel("Brightness (ODOG)")

plt.subplot(2, 2, 4)
plt.bar(
    x=["left", "right", "top", "bottom"], height=targets_LODOG, color=plt.colormaps["tab10"](1)
)
plt.ylim(-2, 2)
plt.axhline(y=0, linestyle="dashed", color="k")
plt.xlabel("Target region")
plt.ylabel("Brightness (LODOG; median)")

plt.show()


# %% [markdown]
# In the horizontal cut, we can see that the model output
# is now greater for the targets in the black phases
# than for the targets in the white phaes
# -- in the same direction as the perceived brightness effect.

# %% [markdown]
# The normalization by global energy in the original ODOG model,
# means this model fails to correctly predict for those images
# where the perceptual effect is present in more localized regions.
# The different regions of the image, which are perceived separately,
# affect each other's normalization.
#
# Instead, the LODOG model implements normalization by _local_ energy estimates.
# This prevents some of this cross-contamination by different image regions.
# As a result, the LODOG model overcomes the challenge to the original model,
# and correctly predicts the direction of effect for the combined stimulus.
