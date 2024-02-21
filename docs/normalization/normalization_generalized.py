# %% [markdown]
# # Generalized -ODOG normalization
# Here we introduce a generalized formulation of the normalization step
# of the -ODOG family of models.
# In this generalized formulation,
# the F-, L-, and ODOG models differ only in parameterization of this step.
# We also show that this formulation is numerically equivalent to the original formulation.
# This is pointed to, and inspired by, a note and Figure 4e
# in Robinson, Hammon & de Sa (2007)

# %% [markdown]
# ## Generalizing normalization steps
# Generally, the -ODOG normalization step consists of three parts:
#
# 1. The _normalizing coefficients_: weighted combinations of all filter outputs
# 2. Energy-calculating (as spatial averaging) of the normalizing coefficients
# 3. Divisive normalization, where a filter output is divided by the energy (2)
#    of its normalizing coefficient (1)
#
# However, the different (F)(L)ODOG models differ in their exact implementation,
# particularly in steps (1) and (2).
# Initially, these differences may seem structural,
# different models using different calculations.
# Yet, as we show here, these can be expressed as parametric differences:
# the same structural components for each step,
# but with different parameter values.
#
# `multyscale` thus implements a _generalized_ version of the -ODOG normalization,
# such that the individual models are different parameterizations of this generalization.
# This formulation is intriguing because
# it exposes the possibility of other parameter values,
# but also because other forms of normalization
# (from other models of early vision)
# may map onto this or an even more generalized version.

# %% Setup
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np

# Import local module
import multyscale

# %% [markdown]
# ## Frontend
# The three models differ in their normalization step,
# but share the same filterbank frontend,
# so it's only necessary to apply this bank
# (and weight the filteroutputs)
# once.

# %% [markdown]
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

# %% Initialize models
ODOG = multyscale.models.ODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
LODOG = multyscale.models.LODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
FLODOG = multyscale.models.FLODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
assert np.array_equal(LODOG.bank.filters, ODOG.bank.filters)
assert np.array_equal(FLODOG.bank.filters, LODOG.bank.filters)

# %% Apply filterbank
# And weight outputs
filters_output = FLODOG.bank.apply(stimulus)
filters_output = FLODOG.weight_outputs(filters_output)

# %% [markdown]
# ## Normalization
# Generally, the -ODOG normalization step consists of three parts:
#
# 1. The _normalizing coefficients_: weighted combinations of all filter outputs
# 2. Energy-calculating (as spatial averaging) of the normalizing coefficients
# 3. Divisive normalization, where a filter output is divided by the energy (2)
#    of its normalizing coefficient (1)
#
# This can be formalized as:
#
# 1. $ n_{o', s', y, x}  := w_{o', s'} \cdot \mathbf{F} $, where:
#   - $\mathbf{F}$ is the whole set of filteroutputs,
#     and each $f_{o, s, y, x}$ is a specific pixel ($y, x$)
#     in the output of filter with specific orientation and spatial scale ($o, s$).
#     Thus, $\mathbf{F}$ is a 4D tensor ($O, S, Y, X$)
#   - $n_{o', s', y, x}$ is a pixel in the _normalizing coefficient_ for filter $o', s'$.
#   - $\mathbf{w}$ is a set of interaction weights, indicating for each $(o', s')$ filter
#     how all other $O, S$ filters combine.
#     Thus, this is a 4D tensor ($O, S, O, S$).
#   - $\cdot$ is a tensor dot-product operation
#
# 2. $ e_{o', s', y, x}  := \sqrt{\mathrm{avg_{xy}}(n_{o, s}^2)} $, where:
#   - $e_{o', s', y, x}$ is a pixel in the _energy estimate_ for filter $o', s'$
#   - $\mathrm{avg_{xy}}$ is a spatial averaging function,
#     taking some average over pixels ($X, Y$) in the _normalizing coefficient_ $n_{o', s'}$.
#
# 3. $ f'_{o', s', y, x} := \frac{f_{o', s', y, x}}{e_{o', s', y, x}} $, where:
#   - $\mathbf{F'}$ is the normalized set of filteroutputs;
#     a 4D tensor of same dimensions and size as $\mathbf{F}$
#
# Combined, this gives:
# 
# $$ f'_{o, s, y, x} := 
#  \frac{f_{o, s, y, x}}
#  {\sqrt{\mathrm{avg_{xy}}((\mathbf{w}\cdot\mathbf{F})^2)}} 
# $$
#
# All three (F)(L)ODOG models can be expressed in this form,
# by changing the implementation of parts (1) and (2), specifically.


# %% [markdown]
# ## Normalizing coefficients
# The first step in normalization is to define
# the normalizing coefficient ($n_{o', s'}$) for each filter ($f_{o, s}$).
# This normalizing coefficient is made up of (a subset of)
# the responses in all $\mathbf{F}$ filter outputs.
# Thus, the tensor of normalizing coefficients $\mathbf{N}$
# contains $O \times S$ 2D ($Y \times X$):
# one normalizing coefficient $n_{o',s'}$ per filter $f_{o',s'}$ to normalize.
#
# In `multyscale`, we calculate these normalizing coefficients
# as a tensor dot-product beween a set of weights $\mathbf{w}$
# and all filter outputs $\mathbf{F}$.
# The 4D tensor ($O, S, O, S$) of interaction weights $\mathbf{w}$
# can be constructed by combined separately defined sets of weights
# for how different orientations interact,
# and how different spatial scale interact.

# %% [markdown]
# In all -ODOG models, filters only normalize other filters with the same orientation,
# i.e., when $o'=o$. Thus, this forms a diagonal matrix of orientation interaction weights

# %% Orientation normalization weights
orientation_norm_weights = multyscale.normalization.orientation_norm_weights(6)
plt.pcolor(
    orientation_norm_weights[::-1, :], cmap="Greens", edgecolors="k", linewidths=1, vmin=0, vmax=1
)
plt.xlabel("Orientation $o'$")
plt.ylabel("Orientation $o$")
plt.show()

# %% [markdown]
# In the base ODOG model, all scales influence all (other) scales equally,
# thus the matrix of scale interaction weights is all $1$s.

# %% Scale normalization weights
scale_norm_weights = multyscale.normalization.scale_norm_weights_equal(7)
plt.pcolor(scale_norm_weights, cmap="Greens", edgecolors="k", linewidths=1, vmin=0, vmax=1)
plt.xlabel("Spatial scale/freq. $s'$")
plt.ylabel("Spatial scale/freq. $s$")
plt.show()

# %% [markdown]
# These normalization weights along each dimension
# are then combined into a single
# $(O' \times S' \times O \times S)$ matrix (tensor) of normalization weights.
#
# This tensor $w_{o',s',o,s}$ can be produced
# using the function `multyscale.normalization.create_normalization_weights()`
# from the separate sets of weights for orientations and scales.
#
# For the example where $(o'=3, s'=4)$,
# this means that all weights $w_{3,4,o,s}=1$ if $o=3$, regardless of $s$.

# %% ODOG Normalization weights
interaction_weights = multyscale.normalization.create_normalization_weights(
    *filters_output.shape[:2], scale_norm_weights, orientation_norm_weights
)

# Visualize weights
fig, axs = plt.subplots(*interaction_weights.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(interaction_weights.shape[:2]):
    axs[o, s].pcolor(
        interaction_weights[interaction_weights.shape[0] - o - 1, s],
        cmap="Greens",
        edgecolors="k",
        linewidths=1,
        vmin=0,
        vmax=1,
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# These interaction weights then are used to combine all filter outputs $\mathbf{F}$
# to create the normalizing coefficients $\mathbf{n}$

# %% Normalizing coefficients
normalizing_coefficients = multyscale.normalization.norm_coeffs(filters_output, interaction_weights)

# Visualize each normalizing coefficient n_{o,s}, i.e.
# the normalizer image for each individual filter f_{o,s}
fig, axs = plt.subplots(*normalizing_coefficients.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalizing_coefficients.shape[:2]):
    axs[o, s].imshow(normalizing_coefficients[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# The LODOG model uses the same interaction weights as the base ODOG model.

# %% [markdown]
# The FLODOG model uses a different set of interaction weights.
# Specifically, it does not weight all spatial scales equally.
# (It does weigh the orientations in the same manner as (L)ODOG).
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
# ### Summary
# In the first step of normalization,
# the construction of normalizing coefficients $\mathbf{n}$
# as weighted combinations of all filter outputs $\mathbf{F}$,
# we can express the three models as using different sets of interaction weights $\mathbf{w}$.
# `multyscale` makes it easy and straightforward to implement these different weights,
# as well as to explore even further with different weights.

# %% [markdown]
# ## Energy estimate through (localized) averaging
# The second step in -ODOG normalization,
# is to use an energy-estimate of, rather than the raw, normalizing coefficients.
# Rather than normalizing by this weighted sum of all filter outputs at each pixel location,
# instead the -ODOG models normalize by
# the _energy_ of the normalizing coefficient.
# Energy here is expressed as the (spatial) root-mean-square of the signal.
#
# 1. Square each ($X \times Y = 1024 \times 1024$) pixel of the normalizing coefficient $n_{o, s}$
# 2. Average \mathrm{avg_{yx}} across pixels
# 3. Square-root of this average
#
# $$ e_{o, s, y, x}  := \sqrt{\mathrm{avg_{yx}}(n_{o, s}^2)} $$
#
# NOTE: this is the _quadratic mean_ of the normalizing coefficient.

# %% [markdown]
# The base ODOG normalization uses the _global image mean_ as the spatial average:
#
# $$ \mathrm{avg_{yx}}(n_{o', s', x, y}) = \frac{1}{YX} n_{o', s', x, y} $$
#
# This results in a single energy estimate for each $(o', s')$.

# %% Global image RMS
ODOG_norm_coeffs = ODOG.norm_coeffs(filters_output)
ODOG_energies = ODOG_norm_coeffs.mean(axis=-1).mean(axis=-1)
print(ODOG_energies.shape)

# Visualise
plt.pcolor(ODOG_energies[::-1, :], cmap="coolwarm", edgecolors="k", linewidths=1)
plt.ylabel("Orientation $o'$")
plt.xlabel("Spatial scale $s'$")
plt.show()

# %% [markdown]
# These energy estimates are then used as the denominator in the divisive normalization step (3)

# %% Divisive normalization
ODOG_outputs = np.ndarray(filters_output.shape)
for o, s in np.ndindex(filters_output.shape[:2]):
    f = filters_output[o, s, ...]
    n = ODOG_energies[o, s]
    ODOG_outputs[o, s] = f / n

# Visualize each normalized f'_{o',s'}
fig, axs = plt.subplots(*ODOG_outputs.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(ODOG_outputs.shape[:2]):
    axs[o, s].imshow(
        ODOG_outputs[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=ODOG_outputs.min(),
        vmax=ODOG_outputs.max(),
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()


# %% [markdown]
# ### Energy estimates as matrices ("images")
# However, it makes conceptual more sense
# to have a denominator that is the same dimensions (Y, X)
# as the filter that is being normalized.

# %% Energies as tensor
ODOG_energies_i = np.tile(ODOG_energies.reshape(6, 7, 1, 1), (1, 1, 1024, 1024))
print(ODOG_energies_i.shape)

# %% [markdown]
# This is equivalent to implementing the averaging in the denominator
# as a multiplication with a matrix where each component is the energy of the normalizer image.
#
# $$ f'_{o',s'} = \frac{f_{o',s'}}{e_{o', s', y,x}} $$
#
# where
#
# $$ e_{o', s', y,x} = \sqrt{\frac{1}{YX}\sum_{y=1}^{Y} \sum_{x=1}^{X}n_{o',s'}^2} $$
#
# This changes the dimensionality of the denominator,
# but not actually the result of the division,
# since image-by-image division is executed pixel-wise.

# %% Divisive normalization
norm_i_outputs = filters_output / ODOG_energies_i

assert np.allclose(ODOG_outputs, norm_i_outputs)


# %% [markdown]
# ### Localized spatial averaging
# Instead of the global image mean,
# the (F)LODOG model uses a Gaussian window $G$ of some with $\sigma$
# to average over pixels,
# giving the _local_ (estimate of) energy:
#
# $$ e_\mathrm{local}(n_{o',s'},\sigma) = \sqrt{G(\sigma) * (n_{o',s',x,y})^2} $$
#
# Thus, difference between (F)LODOG and ODOG normalization is purely in the denominator $e$:
#
# $$
# \begin{aligned}
#     ODOG: e &= \sqrt{\frac{1}{YX}\sum_{y=1}^{Y} \sum_{x=1}^{X} n_{o',s'}^2} \\
# (F)LODOG: e &= \sqrt{G(\sigma) * n_{o',s',x,y}^2} \end{aligned}
# $$
#

# %% Spatial Gaussian
spatial_kernels = np.ndarray(filters_output.shape)
for o, s in np.ndindex(LODOG.window_sigmas.shape[:2]):
    spatial_kernels[o, s, :] = multyscale.normalization.spatial_kernel_gaussian(
        LODOG.bank.x, LODOG.bank.y, LODOG.window_sigmas[o, s]
    )

assert np.array_equal(spatial_kernels, LODOG.spatial_kernels())

# %% [markdown]
# Applying this Gaussian window gives the _local_ (estimate of) of energy

# %% Local RMS
local_energies = np.ndarray(normalizing_coefficients_LODOG.shape)
for o, s in np.ndindex(normalizing_coefficients_LODOG.shape[:2]):
    norm = normalizing_coefficients_LODOG[o, s]
    norm = norm ** 2
    local_avg = multyscale.filters.apply(
        norm, spatial_kernels[o, s], padval=0
    )
    local_energies[o, s] = np.sqrt(local_avg + 1e-6)  # minor offset to avoid negatives/0's

assert np.allclose(local_energies, LODOG.norm_energies(normalizing_coefficients_LODOG, eps=1e-6))

# Visualize each local RMS
fig, axs = plt.subplots(*local_energies.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(local_energies.shape[:2]):
    axs[o, s].imshow(local_energies[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# Since these local energy estimates are not the same across the "image",
# here expressing the estimates as 2D matrices is essential.
# Therefore, it makes sense to also do this for the base ODOG model above,
# to make the comparison clearer.

# %% Divisive normalization
# Since the local energies tensor is the same $(O, S, X, Y)$ shape as the filter outputs
# we can simply divide
LODOG_outputs = filters_output / (local_energies + 1e-6)

# Visualize each normalized f'_{o',s'}
fig, axs = plt.subplots(*LODOG_outputs.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(LODOG_outputs.shape[:2]):
    axs[o, s].imshow(LODOG_outputs[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# ### Implement global image averaging as a spatial filter

# %% [markdown]
# The question then is whether global averaging
#
# $$ \frac{1}{YX}\sum_{y=1}^{Y}\sum_{x=1}^{X} \dots $$
#
# can be reformulated as a convolution with 2D kernel
#
# $$ \mathbf{A}(..) * \dots $$
#
# A convolution is essentially a repeated weighted sum,
# where the weight is the value in the filter
# and this sum is repeated for centering the filter
# on each pixel in the input image.
# If we can construct a kernel $\mathbf{A}$ ensures that for every pixel
# it weights all pixels in the input image by $\frac{1}{YX}$,
# then the output of the convolution is simply the global image average.

# %%
# have doubly sized kernel so that pixel reaches each pixel in the average
spatial_kernel = np.ones((1024 * 2, 1024 * 2)) / 1024**2

img = filters_output[3, 4]
mean_filtered = multyscale.filters.apply(img, spatial_kernel, padval=0)

assert np.allclose(mean_filtered, img.mean())

# %% [markdown]
# With this kind of spatial filtering,
# we run into the question of how to avoid edge artefects.
# For the pixels on the edges of the image,
# the kernel, when centered on one of those pixels,
# will extend beyond the edges of the image.
# In our case, this causes 2 problems:
# - if the kernel were the same size as the input image,
# then some pixels on the _other_ edge of the image
# will no longer fall in our image.
# Thus the spatial kernel $\mathbf{A}$ has $(2Y,2X)$ entries
# to assure that each pixel will take its average from the entire image.
# - what to do at the edges, to ensure that the filter has values to filter?
# We pad all the edges with the value $0$,
# so that no unnecessary information gets included into the averaging.

# %% [markdown]
# The spatial averaging step in the ODOG normalization then becomes:
#
# $$ \mathbf{A} * (\dots) $$
#
# and thus the energy step:
#
# $$ e = \sqrt{\mathbf{A} * n_{o',s'}^2} $$
#

# %% Global image averaging as filter
A = np.ones((1024 * 2, 1024 * 2)) / 1024**2

# calculate energy using the kernel
ODOG_energies_i2 = np.ndarray(filters_output.shape)
for o, s in np.ndindex(filters_output.shape[:2]):
    norm = ODOG_norm_coeffs[o, s, ...] ** 2
    mean = multyscale.filters.apply(norm, A, padval=0)
    ODOG_energies_i2[o, s] = np.sqrt(mean)

# this replicates the just verified normalizing image
assert np.allclose(ODOG_energies_i2, ODOG_energies_i)

# %% [markdown]
# ### Summary
# Thus, we can now reformulate the ODOG normalization as:
#
# $$ f'_{o',s'} = \frac{f_{o',s'}}{\sqrt{\mathbf{A} * (\mathbf{w} \cdot \mathbf{F})^2}} $$
#
# where:
#  - $w_{o', s', o, s} =   \begin{cases}
#      1 & o = o'  \\
#      0 & else
#     \end{cases}$
#  - $\cdot$ is a tensor dot-product
#  - $\mathbf{A} * ...$ means the convolution with our 2D ($Y,X$) kernel
#    that implements global averaging.
#

# %% [markdown]
# ## Generalized formulation
#
# All three (F)(L)ODOG can thus be expressed as parameteric variations
# of the same overall divisive normalization structure:
#
# $$ f'_{o, s, y, x} := \frac{f_{o, s, y, x}}{\sqrt{\mathbf{A} * (\mathbf{w} \cdot \mathbf{F})^2}} $$
#
# Both $\mathbf{w}$ and $\mathbf{A}$ depend on the specific model:
# - $\mathbf{w}$ is the same for LODOG and ODOG, where
#    $ w_{o', s', o, s} =   \begin{cases}
#      1 & o = o'  \\
#      0 & else
#     \end{cases} $
#   For FLODOG, the $ w_{o', s'} $ also depends on the relative index of $ (s', s) $
# - $ \mathbf{A} $ is Gaussian filter $\mathbf{G(\sigma)}$ for LODOG and FLODOG,
#   where $\sigma = k$ and $\sigma = ks$, respectively.
#   For ODOG, $ \mathbf{A} $ is a (larger) constant filter.
#   However, this could even be understood as an infinite-width Gaussian $ \mathbf{G(\infty)} $

# %% [markdown]
# ### Pseudo-implementation


# %%
def divisive_normalization(filter_output, norm_energy):
    return filter_output / norm_energy


def norm_energy(norm_coeff, spatial_kernel):
    norm = norm_coeff**2
    spatial_average = multyscale.filters.apply(norm, spatial_kernel, padval=0)
    energy = np.sqrt(spatial_average)
    return energy


def norm_coeff(filter_outputs, normalization_weights):
    coeffs = np.zeros_like(filter_outputs)
    for o, s in np.ndindex(filter_outputs.shape[:2]):
        weights = normalization_weights[o, s]

        # Tensor dot: multiply filters_output by weights, then sum over axes [0,1]
        coeff = np.tensordot(filter_outputs, weights, axes=([0, 1], [0, 1]))

        # Accumulate
        coeffs[o, s, ...] = coeff

    return coeffs


def spatial_kernel_ODOG(shape):
    kernel = np.ones([dim * 2 for dim in shape])
    kernel /= kernel.sum()
    return kernel


def spatial_kernel_LODOG(x, y, sigmas=[1.0, 1.0]):
    kernel = multyscale.filters.gaussian2d(x, y, sigmas)
    kernel /= kernel.sum()
    return kernel
