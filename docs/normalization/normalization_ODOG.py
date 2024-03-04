# %% [markdown]
# # ODOG normalization
# This Tutorial describes the rationale behind
# the normalization step of the original ODOG model (Blakeslee & McCourt, 1997),
# and its implementation in `multyscale`.


# %% Setup
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np

# Import local module
import multyscale

# %% [markdown]
# ## Frontend


# %% [markdown]
# ### Example stimulus
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
# ### Filterbank
# The -DOG family of models starts with a _multiscale spatial filtering_ frontend.
# This consists of a set of filters, $\mathbf{F}$
# which span a range of spatial scales $S$
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

# %% Frontend filterbank of ODOG implementation by Robinson et al. (2007)
filterbank = multyscale.filterbanks.RHS2007(filtershape=stimulus.shape, visextent=visextent)

# Get parameters
print(f"{filterbank.filters.shape[0]} orientations, {filterbank.filters.shape[1]} spatial scales")

# Visualise filterbank
fig, axs = plt.subplots(*filterbank.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(*filterbank.shape[:2]):
    axs[o, s].imshow(filterbank.filters[o, s, ...], cmap="coolwarm", extent=visextent)
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
filters_output = filterbank.apply(stimulus)

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
# ### Recombination is not enough
# Simply recombining the outputs of these filters does not give rise to any effects;
# it merely gives a lossy reconstruction of the original stimulus,
# since the filters act as a decomposition.

# %% Recombine
recombined_outputs = np.sum(filters_output, axis=(0, 1))

plt.subplot(1, 2, 1)
plt.imshow(recombined_outputs, cmap="gray", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(1, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], recombined_outputs.shape[1])[256:768],
    recombined_outputs[512, 256:768],
    color="black",
)
plt.show()

# %% [markdown]
# In the horizontal cut, we can see that the model output
# is still approximately equal for the two gray target patches in the stimulus image.


# %% [markdown]
# ### Frontend: weighting filter(outputs) according to CSF
# In the -ODOG models, the filter outputs are weighted according to the CSF,
# that is, higher frequencies (smaller spatial scales),
# are weighted more strongly:
# "The seven spatial frequency filters [are weighted] across frequency
#  using a power function with a slope of 0.1"
#
# The weights can be created using `multyscale.filterbank.scale_weights`,
# and applied using `multyscale.filterbanks.weight_oriented_multiscale_outputs`.

# %% Weight individual filter(outputs) according to spatial size (frequency)
center_sigmas = [center[0] for (center, s) in filterbank.sigmas]
weights = multyscale.filterbanks.scale_weights(center_sigmas, slope=0.1)

weighted_outputs = multyscale.filterbanks.weight_oriented_multiscale_outputs(
    filters_output, weights
)

# Visualise weighted filter outputs
fig, axs = plt.subplots(*weighted_outputs.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(weighted_outputs.shape[:2]):
    axs[o, s].imshow(
        weighted_outputs[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=weighted_outputs.min(),
        vmax=weighted_outputs.max(),
    )
fig.supxlabel("Spatial scale/freq. $s$")
fig.supylabel("Orientation $o$")
plt.show()

# %% [markdown]
# This does produce some effect, however, not the kind we wish to model:

# %% Readout
recombined_outputs = np.sum(weighted_outputs, axis=(0, 1))

plt.subplot(1, 2, 1)
plt.imshow(recombined_outputs, cmap="gray", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(1, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], recombined_outputs.shape[1])[256:768],
    recombined_outputs[512, 256:768],
    color="black",
)
plt.show()

# %% [markdown]
# In the horizontal cut, we can see that the model output
# is actually _higher_ for the right patch, compared to the left patch,
# which is _opposite_ from the perceived brightness difference!


# %% [markdown]
# ## Normalization
# The predictive power of the -ODOG models comes from their normalization step,
# in which information from all the filters regulates the activity of each filter output.
#
# Each filter(output) $f_{o',s'}$ gets divided by
# a _normalizing coefficient_ $n_{o',s'}$,
# i.e., for each filter $f_{o',s'}$, the normalized $f'$ is
#
# $$f'_{o',s'} := \frac{f_{o',s'}}{n_{o',s'}}$$
#
# The normalizing coefficient $n_{o', s'}$ is made up of (a subset of)
# the responses in all $\mathbf{F}$ filter outputs.
# Thus, the tensor of normalizing coefficients $\mathbf{N}$
# contains $6 \times 7$ 2D ($X \times Y$):
# one normalizing coefficient $n_{o',s'}$ per filter $f_{o',s'}$ to normalize.

# %% [markdown]
# ### Combine filter outputs into normalizing coefficients
#
# In the original ODOG specification
# a filter only gets normalized by the filters with the same orientation.
# Thus, the normalizing coefficient $n_{o',s'}$
# is a combination of only those $f_s$ with the same orientation ($o=o'$):
#
# $$n_{o',s', x, y} := \sum_{s=1}^{S} f_{o',s,x,y}$$

# %% Normalizing coefficients
norm_coeffs = np.zeros_like(weighted_outputs)
for o_prime, s_prime in np.ndindex(weighted_outputs.shape[:2]):  # for each filter to normalize
    for o, s in np.ndindex(weighted_outputs.shape[:2]):  # loop over all filters
        if o == o_prime:  # same orientation
            norm_coeffs[o_prime, s_prime] += weighted_outputs[
                o, s, :
            ]  # add this filter to normalizing coefficient

# Plot each normalizing coefficient n_{o,s},
# i.e., for each individual filter f_{o,s}
fig, axs = plt.subplots(*norm_coeffs.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(norm_coeffs.shape[:2]):
    axs[o, s].imshow(
        norm_coeffs[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=norm_coeffs.min(),
        vmax=norm_coeffs.max(),
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# NOTE that all normalizing coefficients within a row are identical,
# i.e., all filters of an orientation get the same normalizing coefficient in the ODOG model.
# This makes sense,
# since each filter $f_{o',...}$ gets normalized by all filters of that same orientation $o=o'$:
#
# $$n_{o',1} = n_{o',2} = ... = n_{o',7} = \sum_{s=1}^{S} f_{o',s,x,y}$$


# %% [markdown]
# We can alternatively specify the combination
# as a weighted sum over all $\mathbf{F}$ such that
# $$n_{o',s',x,y}=\sum_{o=1}^{O}\sum_{s=1}^{S} w_{o',s',o,s}f_{o,s,x,y}$$
# where the weight depends on whether $o$:
# $$w_{o',s',o,s} = \begin{cases}
#       1 & o = o'  \\
#       0 & else
# \end{cases}$$
#
# Thus, $\mathbf{W}$ is a $O \times S$ set of $O \times S$ weights:
# for each $f_{o',s'}$ filter, we need to define $O \times S$ weights
# for whether each filter normalizes this one.

# %% Define normalization weights
normalization_weights = np.ndarray(weighted_outputs.shape[:2] * 2)
for o_prime, s_prime in np.ndindex(weighted_outputs.shape[:2]):
    for o, s in np.ndindex(weighted_outputs.shape[:2]):
        if o == o_prime:
            normalization_weights[o_prime, s_prime, o, s] = 1  # /filters_output.shape[1]
        else:
            normalization_weights[o_prime, s_prime, o, s] = 0

# Visualize weights
fig, axs = plt.subplots(*normalization_weights.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalization_weights.shape[:2]):
    axs[o, s].pcolor(
        normalization_weights[normalization_weights.shape[0] - o - 1, s],
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
# With a tensor (matrix) dot-product,
# these weights can be used to combine filter outputs into normalizing images

# %% Normalizing images as weighted combination (tensor dot-product) of filter outputs
normalizing_coefficients = np.tensordot(
    normalization_weights, weighted_outputs, axes=([0, 1], [0, 1])
)

# Visualize each normalizing coefficient n_{o,s},
# i.e., for each individual filter f_{o,s}
fig, axs = plt.subplots(*normalizing_coefficients.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalizing_coefficients.shape[:2]):
    axs[o, s].imshow(normalizing_coefficients[o, s], cmap="coolwarm", extent=visextent)

# NOTE that these normalizing coefficients are identical the ones above:
assert np.allclose(normalizing_coefficients, norm_coeffs)

# %% [markdown]
# In `multyscale`, such a matrix (tensor) of normalization weights
# can be easily and expressively created:
# by specifying the weights along each dimension (orientation, scale, etc.)
# and combining these into the single tensor.
#
# The function `multyscale.normalization.scale_norm_weights_equal()`
# defines how the $S$ spatial scales affect each other
# as a $(S \times S)$ matrix of weights:
# for each scale-to-be-normalized $s'$,
# it produces the weight given to each other scale $s$.
#
# In the ODOG model, all scales influence all (other) scales equally,
# thus this matrix is all $1$s.

# %% Scale normalization weights
scale_norm_weights = multyscale.normalization.scale_norm_weights_equal(7)
plt.pcolor(scale_norm_weights, cmap="Greens", edgecolors="k", linewidths=1, vmin=0, vmax=1)
plt.xlabel("Spatial scale/freq. $s'$")
plt.ylabel("Spatial scale/freq. $s$")
plt.show()

# %% [markdown]
# The function `multyscale.normalization.orientation_norm_weights()` similarly
# defines how the $O$ spatial scales affect each other,
# as a $(O \times O)$ matrix of weights:
# for each orientation-to-be-normalized $o'$,
# it produces the $O$ weights given to all other scales $o$.
#
# In the -ODOG models, filters only normalize other filters with the same orientation,
# i.e., when $o'=o$. Thus, this forms a diagonal matrix

# %% Orientation normalization weights
orientation_norm_weights = multyscale.normalization.orientation_norm_weights(6)
plt.pcolor(
    orientation_norm_weights[::-1, :], cmap="Greens", edgecolors="k", linewidths=1, vmin=0, vmax=1
)
plt.xlabel("Orientation $o'$")
plt.ylabel("Orientation $o$")
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

# %% Normalization weights
norm_weights = multyscale.normalization.create_normalization_weights(
    *weighted_outputs.shape[:2], scale_norm_weights, orientation_norm_weights
)
# NOTE that these are identical to the weights $w$ defined above.
assert np.array_equal(norm_weights, normalization_weights)

# Visualize weights
fig, axs = plt.subplots(*norm_weights.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(norm_weights.shape[:2]):
    axs[o, s].pcolor(
        norm_weights[norm_weights.shape[0] - o - 1, s],
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
# `multyscale` also provides convenience functionality
# for constructing the normalizing coefficients
# from such weights, and the filter outputs to be normalized.
#
# The function `multyscale.normalization.norm_coeffs()` creates these coefficients,
# and note that these are identical to the $N$ constructed above.

# %% Identical normalizing coefficients
norm_coeffs = multyscale.normalization.norm_coeffs(weighted_outputs, normalization_weights)
assert np.allclose(norm_coeffs, normalizing_coefficients)

# Visualize each normalizing coefficient n_{o,s},
# i.e. for each individual filter f_{o,s}
fig, axs = plt.subplots(*normalizing_coefficients.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalizing_coefficients.shape[:2]):
    axs[o, s].imshow(normalizing_coefficients[o, s], cmap="coolwarm", extent=visextent)
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# ### Normalize by energy
# Rather than normalizing by this weighted sum of all filter outputs at each pixel location,
# instead the -ODOG models normalize by
# the _energy_ of the normalizing coefficient.
# Energy here is expressed as the (spatial) root-mean-square of the signal.
#
# 1. Square each ($X \times Y = 1024 \times 1024$) pixel of the combined normalizing coefficient
# 2. Mean across all ($X \times Y = 1024 \times 1024$) pixels
# 3. Square-root of this mean
#
# $$ RMS(n_{o',s'}) = \sqrt{\frac{1}{XY}\sum_{x=1}^{X}\sum_{y=1}^{Y}(n_{o',s',x,y})^2} $$
#
# NOTE: this is the _quadratic mean_ of the normalizing coefficient.
#
# This results in a single value for each $f_{o', s'}$ filter-to-be-normalized.

# %% Global image RMS
normalization_RMSs = np.ndarray(normalizing_coefficients.shape[:2])
for o, s in np.ndindex(normalizing_coefficients.shape[:2]):
    normalization_RMSs[o, s] = np.sqrt((normalizing_coefficients[o, s] ** 2).mean())

# Visualise
plt.pcolor(normalization_RMSs[::-1, :], cmap="coolwarm", edgecolors="k", linewidths=1)
plt.ylabel("Orientation $o'$")
plt.xlabel("Spatial scale $s'$")
plt.show()

# %% [markdown]
# This heatmap represents the RMS value of the normalizing coefficient
# that each filter gets normalized by.

# %% [markdown]
# ### Divisive normalization
# These values, the energy (spatial RMS) of the normalizing coefficients,
# form the denominator of the divisive normalization.
#
# Thus, the normalized filter output $f'_{o', s'}$
# is calculating by dividing each filter(output) $f_{o',s'}$
# by the energy of the normalizing coefficient $n_{o',s'}$:
# $$f'_{o',s'} = \frac{f_{o',s'}}{RMS(n_{o',s'})}$$

# %% Divisive normalization
normalized_outputs = np.ndarray(weighted_outputs.shape)
for o, s in np.ndindex(weighted_outputs.shape[:2]):
    f = weighted_outputs[o, s, ...]
    n = normalization_RMSs[o, s]
    normalized_outputs[o, s] = f / n

# Visualize each normalized f'_{o',s'}
fig, axs = plt.subplots(*normalized_outputs.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalized_outputs.shape[:2]):
    axs[o, s].imshow(
        normalized_outputs[o, s],
        cmap="coolwarm",
        extent=visextent,
        vmin=normalized_outputs.min(),
        vmax=normalized_outputs.max(),
    )
fig.supxlabel("Spatial scale/freq. $s'$")
fig.supylabel("Orientation $o'$")
plt.show()

# %% [markdown]
# In this visualisation of the normalized outputs,
# again, rows differ in orientation of the filter (being normalized)
# and columns differ in the spatial scale of the filter (being normalized).

# %% [markdown]
# Thus, the full normalization schema of ODOG can be formulated as:
# $$ F' =
# \frac{f_{o',s',x,y}}{\sqrt{\frac{1}{XY}\sum_{y=1}^{Y} \sum_{x=1}^{X}(\sum_{o=1}^{O}\sum_{s=1}^{S} {w_{o',s',o,s} f_{o,s,x,y})^2}}}
# = \frac{f_{o',s',x,y}}{\sqrt{\mathbf{avg}((\mathbf{W} \cdot \mathbf{F})^2)}}
# $$
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
recombined_outputs = np.zeros(normalized_outputs.shape[-2:])
for o, s in np.ndindex(normalized_outputs.shape[:2]):
    recombined_outputs += normalized_outputs[o, s]

# Visualize model output
plt.subplot(1, 2, 1)
plt.imshow(recombined_outputs, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(1, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], recombined_outputs.shape[1])[256:768],
    recombined_outputs[512, 256:768],
    color="black",
)
plt.show()

# %% [markdown]
# In the horizontal cut, we can see that the model output
# is now greater for the target patch on the left side of the stimulus,
# than for the patch on the right side
# -- in the same direction as the perceived brightness effect.

# %% [markdown]
# Thus, the normalization step is crucial for the proper functioning of the ODOG model.
# Robinson, Hammon, & de Sa (2007) also demonstrate this,
# by comparing the full ODOG model with a version without normalization (UNODOG).
# The UNODOG model fails to predict the direction of effect more often than the full ODOG model
# (Table 2).
