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
# 1. $ n_{o, s, y, x}  := w_{o, s} \cdot \mathbf{F} $, where:
#   - $n_{o, s, y, x}$ is a pixel in the _normalizing coefficient_ for filter $o, s$.
#   - $\mathbf{F}$ is the whole set of filteroutputs,
#     and each f_{o, s, y, x} is a specific pixel ($y, x$)
#     in the output of filter with specific orientation and spatial scale ($o, s$).
#     Thus, $\mathbf{F}$ is a 4D tensor ($O, S, Y, X$)
#   - $\mathbf{w}$ is a set of interaction weights, indicating for each $(o, s)$ filter
#     how all other $O, S$ filters combine.
#     Thus, this is a 4D tensor ($O, S, O, S$).
#   - $\cdot$ is a tensor dot-product operation
#
# 2. $ e_{o, s, y, x}  := \sqrt{\mathrm{avg_{xy}}(n_{o, s}^2)} $, where:
#   - $e_{o, s, y, x} is a pixel in the _energy estimate_ for filter $o, s$
#   - $\mathrm{avg_{xy}}$ is a spatial averaging function,
#     taking some average over pixels ($X, Y$) in the _normalizing coefficient_ $n_{o, s}$.
#
# 3. $ f'_{o, s, y, x} := \frac{f_{o, s, y, x}}{e_{o, s, y, x}} $, where:
#   - $\mathbf{F'}$ is the normalized set of filteroutputs;
#     a 4D tensor of same dimensions and size as $\mathbf{F}$
#
# Combined, this gives:
# $$
# f'_{o, s, y, x} :=
# \frac{f_{o, s, y, x}}
# {\sqrt{\mathrm{avg_{xy}}(
# (\mathbf{w}\cdot\mathbf{F})^2
# )}}
# $$
#
# All three (F)(L)ODOG models can be expressed in this form,
# by changing the implementation of parts (1) and (2), specifically.


# %% [markdown]
# ## Normalizing coefficients
# The first step in normalization is to define
# the normalizing coefficient ($n_{o, s}$) for each filter ($f_{o, s}$).
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
# this means that all weights $w_{3,4,o,s}=1$ if $o==3$, regardless of $s$.

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
normalizing_coefficients = multyscale.normalization.normalizers(filters_output, interaction_weights)

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
normalizing_coefficients_LODOG = LODOG.normalizers(filters_output)
normalizing_coefficients_FLODOG = FLODOG.normalizers(filters_output)

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
# ### ODOG
# The ODOG normalization step can be formulated as:
# $$
# f'_{o',s',x,y} = \frac{f_{o',s',x,y}}
# {\sqrt{\frac{1}{XY}\sum_{y=1}^{Y} \sum_{x=1}^{X}n_{o',s'}^2}}
# $$
# where $w_{o', s', o, s} =   \begin{cases}
#       1 & o = o'  \\
#       0 & else
#    \end{cases}$
# and
# $$
# n_{o',s'} = \sum_{o=1}^{O}\sum_{s=1}^{S} {w_{o',s',o,s} f_{o,s,x,y}}
# $$

# %%
ODOG_outputs = ODOG.normalize_outputs(filters_output)

# %% [markdown]
# #### Can also divide by 2D "image" arrays
# In the original formulation of the ODOG normalization step,
# the denominator is a single scalar
# -- the root mean square of the normalizer image $n_{o',s'}$

# %%
ODOG_normalizers = ODOG.normalizers(filters_output)
ODOG_RMS = ODOG.normalizers_to_RMS(ODOG_normalizers)
print(ODOG_RMS.shape)

# %% [markdown]
# However, it makes conceptual more sense
# to have a denominator that is the same 2 dimensions
# as the filter that is being normalized.

# %%
i_RMS = np.tile(ODOG_RMS.reshape(6, 7, 1, 1), (1, 1, 1024, 1024))

norm_i_outputs = np.ndarray(filters_output.shape)
for o, s in np.ndindex(filters_output.shape[:2]):
    norm_i_outputs[o, s] = filters_output[o, s, ...] / i_RMS[o, s]

assert np.allclose(ODOG_outputs, norm_i_outputs)

# %% [markdown]
# This is equivalent to implementing the averaging in the denominator
# as a multiplication with a matrix where each component is the RMS of the normalizer image.
#
# $$
# f'_{o',s'} = \frac{f_{o',s'}}{\sqrt{\mathbf{M}}}
# $$
# where
# $$
# m_{x,y} = \frac{1}{XY}\sum_{y=1}^{Y} \sum_{x=1}^{X}n_{o',s'}^2
# $$
#
# This changes the dimensionality of the denominator,
# but not actually the result of the division,
# since image-by-image division is executed pixel-wise.

# %% [markdown]
# ### (F)LODOG
# The (F)LODOG normalization step can be formulated as:
# $$
# f'_{o',s'} = \frac{f_{o',s'}}
# {\sqrt{G(\sigma) * (\sum_{o=1}^{O}\sum_{s=1}^{S} w_{o',s',o,s}f_{o,s})^2}}
# $$
# where $w_{o', s', o, s} =   \begin{cases}
#       1 & o = o'  \\
#       0 & else
#    \end{cases}$
# and $G({\sigma}) * ...$ means convolution with a 2D ($X,Y$) kernel
# -- in this case, a Gaussian with standard deviation $\sigma$ in both directions.
#
# NOTE: FLODOG and LODOG differ just in the _values_ for $\sigma$ and $\mathbf{W}$

# %%
LODOG = multyscale.models.LODOG_RHS2007(shape=stimulus.shape, visextent=visextent)
FLODOG = multyscale.models.FLODOG_RHS2007(shape=stimulus.shape, visextent=visextent)

# %%
LODOG_outputs = LODOG.normalize_outputs(filters_output)
FLODOG_outputs = FLODOG.normalize_outputs(filters_output)

# %% [markdown]
# ### Difference(s)
# The difference between (F)LODOG and ODOG normalization is purely in the denominator $\mathbf{\sqrt{M}}$:
# $$
# \begin{aligned}
# ODOG: \mathbf{M} &= \frac{1}{XY}\sum_{y=1}^{Y} \sum_{x=1}^{X} n_{o',s'}^2 \\
#
# (F)LODOG: \mathbf{M} &= G(\sigma) * n_{o',s'}^2
# \end{aligned}
# $$

# %% [markdown]
# For ODOG and LODOG, the weights $\mathbf{W}$ are identical,
# and therefore so are the normalizer images $\mathbf{N}$.

# %%
assert np.allclose(ODOG.normalizers(filters_output), LODOG.normalizers(filters_output))

# %% [markdown]
# ### Implement global image averaging as a spatial filter

# %% [markdown]
# The question then is whether global averaging
# $$
# \mathbf{M} = \frac{1}{XY}\sum_{y=1}^{Y}\sum_{x=1}^{X} ...
# $$
# can be reformulated as a convolution with 2D kernel
# $$
# \mathbf{A}(..) * ...
# $$
# A convolution is essentially a repeated weighted sum,
# where the weight is the value in the filter
# and this sum is repeated for centering the filter
# on each pixel in the input image.
# If we can construct a kernel $\mathbf{A}$ ensures that for every pixel
# it weights all pixels in the input image by $\frac{1}{XY}$,
# then the output of the convolution is simply the global image average.

# %%
# have doubly sized kernel so that pixel reaches each pixel in the average
A = np.ones((1024 * 2, 1024 * 2)) / 1024**2

img = filters_output[3, 4]
mean_filtered = multyscale.filters.apply(img, A, padval=0)

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
# Thus the spatial kernel $\mathbf{A}$ has $(2X,2Y)$ entries
# to assure that each pixel will take its average from the entire image.
# - what to do at the edges, to ensure that the filter has values to filter?
# We pad all the edges with the value $0$,
# so that no unnecessary information gets included into the averaging.

# %% [markdown]
# The spatial averaging step in the ODOG normalization then becomes:
# $$
# \mathbf{M} = \mathbf{A} * (...)
# $$
# and thus the RMS step:
# $$
# \sqrt{\mathbf{M}} = \sqrt{\mathbf{A} * n_{o',s'}^2}
# $$

# %%
A = np.ones((1024 * 2, 1024 * 2)) / 1024**2

# calculate RMS using the kernel
i_RMS2 = np.ndarray(filters_output.shape)
for o, s in np.ndindex(filters_output.shape[:2]):
    norm = ODOG_normalizers[o, s, ...] ** 2
    mean = multyscale.filters.apply(norm, A, padval=0)
    i_RMS2[o, s] = np.sqrt(mean)

# this replicates the just verified normalizing image
assert np.allclose(i_RMS2, i_RMS)

# %% [markdown]
# Thus, we can now reformulate the ODOG normalization as:
# $$
# f'_{o',s'} = \frac{f_{o',s'}}
# {\sqrt{\mathbf{A} * (\sum_{o=1}^{O}\sum_{s=1}^{S} w_{o',s',o,s}f_{o,s})^2}}
# $$
# where $w_{o', s', o, s} =   \begin{cases}
#       1 & o = o'  \\
#       0 & else
#    \end{cases}$
# and $\mathbf{A} * ...$ means the convolution with our 2D ($X,Y$) kernel
# that implements global averaging.

# %%
A = np.ones((1024 * 2, 1024 * 2)) / 1024**2

normed = np.ndarray(filters_output.shape)
for o, s in np.ndindex(filters_output.shape[:2]):
    norm = ODOG_normalizers[o, s, ...] ** 2
    mean = multyscale.filters.apply(norm, A, padval=0)
    normed[o, s] = filters_output[
        o,
        s,
    ] / np.sqrt(mean)

assert np.allclose(normed, ODOG_outputs)


# %% [markdown]
# ## Testing
# %%
def divisive_normalization(filter_output, norm_coeff):
    return filter_output / norm_coeff


def norm_coeff(normalizer, spatial_kernel):
    norm = normalizer**2
    spatial_average = multyscale.filters.apply(norm, spatial_kernel, padval=0)
    coeff = np.sqrt(spatial_average + 1e-6)
    return coeff


def spatial_kernel_ODOG(x, y):
    return np.ones([dim * 2 for dim in x.shape]) / np.prod(x.shape)


def spatial_kernel_LODOG(x, y, sigmas=[0, 0]):
    kernel = multyscale.filters.gaussian2d(x, y, sigmas)
    kernel /= kernel.sum()
    return kernel


# %% [markdown]
# ### Unit tests

# %%
ODOG_kernel = spatial_kernel_ODOG(ODOG.bank.x, ODOG.bank.y)
assert np.all(ODOG_kernel == A)

# %%
LODOG_kernel = spatial_kernel_LODOG(
    ODOG.bank.x, ODOG.bank.y, sigmas=(LODOG.window_sigma, LODOG.window_sigma)
)
LODOG_kernels = multyscale.normalization.spatial_avg_windows_gaussian(
    ODOG.bank.x, ODOG.bank.y, LODOG.window_sigmas
)

assert np.allclose(LODOG_kernel, LODOG_kernels[0, 0])

# %%
img = filters_output[3, 4]
kernel = spatial_kernel_ODOG(ODOG.bank.x, ODOG.bank.y)
mean_filtered = multyscale.filters.apply(img, A, padval=0)

assert np.allclose(mean_filtered, img.mean())

# %%
coeffs = np.ndarray(filters_output.shape)
for o, s in np.ndindex(filters_output.shape[:2]):
    coeffs[o, s, ...] = norm_coeff(ODOG_normalizers[o, s, ...], A)

assert np.allclose(coeffs, i_RMS2)

# %% [markdown]
# ### ODOG

# %%
ODOG_normalizers = ODOG.normalizers(filters_output)

ODOG_outputs = ODOG.normalize_outputs(filters_output)

kernel = spatial_kernel_ODOG(ODOG.bank.x, ODOG.bank.y)

new_normed = np.ndarray(filters_output.shape)
for o_prime, s_prime in np.ndindex(filters_output.shape[:2]):
    normalizer = ODOG_normalizers[o_prime, s_prime]
    coeff = norm_coeff(normalizer, kernel)
    new_normed[o_prime, s_prime] = divisive_normalization(filters_output[o_prime, s_prime], coeff)

assert np.allclose(new_normed, ODOG_outputs)

# %% [markdown]
# ### LODOG

# %%
ODOG_normalizers = ODOG.normalizers(filters_output)

LODOG_outputs = LODOG.normalize_outputs(filters_output)

kernel = spatial_kernel_LODOG(
    ODOG.bank.x, ODOG.bank.y, sigmas=(LODOG.window_sigma, LODOG.window_sigma)
)

new_normed = np.ndarray(filters_output.shape)
for o_prime, s_prime in np.ndindex(filters_output.shape[:2]):
    normalizer = ODOG_normalizers[o_prime, s_prime]
    coeff = norm_coeff(normalizer, kernel)
    new_normed[o_prime, s_prime] = divisive_normalization(filters_output[o_prime, s_prime], coeff)

assert np.allclose(new_normed, LODOG_outputs)

# %% [markdown]
# ### FLODOG

# %%
FLODOG_normalizers = FLODOG.normalizers(filters_output)

FLODOG_outputs = FLODOG.normalize_outputs(filters_output)

kernels = multyscale.normalization.spatial_avg_windows_gaussian(
    ODOG.bank.x, ODOG.bank.y, FLODOG.window_sigmas
)

new_normed = np.ndarray(filters_output.shape)
for o_prime, s_prime in np.ndindex(filters_output.shape[:2]):
    normalizer = FLODOG_normalizers[o_prime, s_prime]
    coeff = norm_coeff(normalizer, kernels[o_prime, s_prime]) + 1e-6
    new_normed[o_prime, s_prime] = divisive_normalization(filters_output[o_prime, s_prime], coeff)

assert np.allclose(new_normed, FLODOG_outputs)
