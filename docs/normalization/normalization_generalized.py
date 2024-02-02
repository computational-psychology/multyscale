# %% [markdown]
# # Generalized -ODOG normalization
# Here we introduce a generalized way of formulating the normalization step of the -ODOG family of models,
# such that the F-, L-, and ODOG models differ only in parameterization of this step,
# and show that this formulation is numerically equivalent to the original formulation.

# %%
# Third party libraries
import numpy as np
from PIL import Image

# Import local module
import multyscale

# %% [markdown]
# ## Frontend

# %%
# %% Load example stimulus
stimulus = np.asarray(Image.open("example_stimulus.png").convert("L"))

# %% Parameters of image
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %%
ODOG = multyscale.models.ODOG_RHS2007(shape=stimulus.shape, visextent=visextent)

# %%
# Frontend filterbank of ODOG implementation by Robinson et al. (2007)
O, S = ODOG.bank.filters.shape[:2]

# %%
# Filter the example stimulus
filters_output = ODOG.bank.apply(stimulus)

# %%
# Weight individual filter(outputs) according to spatial size (frequency)
filters_output = ODOG.weight_outputs(filters_output)

# %% [markdown]
# This preamble defines a filterbank output which is not already normalized by any function.

# %% [markdown]
# ## Normalization steps

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
# ## Generalization

# %% [markdown]
# The -ODOG normalization step can now generally be formulated as:
# $$ f'_{o',s'} =
# \frac{f_{o',s'}}
# {\sqrt{\mathbf{A}*(\sum_{o=1}^{O}\sum_{s=1}^{S} {w_{o',s',o,s} f_{o,s,x,y})^2}}}
# $$
# where
# $$
# w_{o', s', o, s} =
# \begin{cases}
#       1 & o = o' & (L)ODOG  \\
#       G(\sigma, s', s) & o = o' & FLODOG \\
#       0 & else
# \end{cases}
# $$
# and
# $$
# \mathbf{A} =
# \begin{cases}
#       \frac{1}{XY} & ODOG  \\
#       G(\sigma) & LODOG \\
#       G(\sigma * s') & FLODOG
# \end{cases}
# $$

# %% [markdown]
# This identifies two parts in which the three models differ,
# which we name as follows:
# - the _normalization weights_ $\mathbf{W}$
# - the _spatial averaging kernel_ $\mathbf{A}$
#
# These are both featured in the _normalization coefficient_
# that is the denominator of the _divisive normalization_ function.

# %% [markdown]
#
# An implementation of this formulation can be found here:


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
# ## Testing

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
