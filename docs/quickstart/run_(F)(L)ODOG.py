# %% [markdown]
# # Run an existing (-ODOG) model
# This Recipe describes how to run the existing -ODOG models
# (Blakeslee & McCourt, 1999; Robinson, Hammon, & de Sa, 2007)
# as implemented in the module.

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
# ## (F)(L)ODOG models
# The ODOG model (Blakeslee & McCourt, 1997),
# and later derivations LODOG and FLODOG (Robinson, Hammon, & de Sa, 2007)
# are _image-computable model_s of brightness perception.
# As image computable models, they take any arbitrary image-array (2D) as input,
# and output another 2D array
# where each pixel is the predicted perceived brightness for the corresponding pixel in the input.
#
# This recipe demonstrates how to apply these models in one integrated run,
# as well as breaks it down into its constituent components
# and executes these step-wise.
#
# These models are implement in `multyscale.models`,
# where the ODOG model is implemented as the class `ODOG_RHS2007`
# (note that this implementation mimics that of Robinson, Hammon, & de Sa, 2007,
# and deviates slightly from the Blakeslee & McCourt original implementation).
# The specification of the model depends on the image resolution and visual extent of the model,
# so requires these as constructor arguments.


# %% Create model
model_ODOG = multyscale.models.ODOG_RHS2007(stimulus.shape, visextent)

# %% [markdown]
# ## Running the ODOG model
# The most straightforward way of running the model,
# is by calling the `apply()` method

# %% Apply model
output_ODOG = model_ODOG.apply(stimulus)

# Visualise output
plt.subplot(1, 2, 1)
plt.imshow(output_ODOG, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(1, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    output_ODOG[512, 256:768],
    color="black",
)
plt.show()

# %% [markdown]
# In the horizontal cut on the right, we can see that the model output
# is greater for the target patch on the left side of the stimulus,
# than for the patch on the right side
# -- also indicated in the output image by the warmer color on the left than on the right.
# This is in the same direction as the perceived brightness effect.


# %% [markdown]
# ### Readout
# From this output 2D-array,
# which represents the (predicted) brightness at each pixel location in the original input image,
# we can readout predicted brightness for target regions.
#
# To do this, we use a _mask_-array:
# a 2D-array with the same $Y \times X$ shape as the input image
# where each pixel is assigned to a region of interest.
# At each pixel location,
# an integer index indicates which region of interest it belongs to.

# %% Mask
mask = np.load("example_stimulus_mask.npy")

fig, (ax_im, ax_mask) = plt.subplots(1, 2)
ax_im.imshow(stimulus, cmap="gray", extent=visextent)
ax_mask.imshow(mask, extent=visextent)
plt.show()

# %% [markdown]
# By indexing only those pixels in the image
# belong to a given masked region,
# we can average (median) the intensity value in this region.

# %% Extract targets intensities
target_intensities = []
for idx in np.unique(mask.astype(int)):
    if idx > 0:
        target_intensities.append(np.median(stimulus[mask == idx]))

# Visualize
plt.bar(x=["left", "right"], height=target_intensities, color="k")
plt.ylim([0, 1])
plt.xlabel("Target region")
plt.ylabel("Intensity (median)")
plt.show()

# %% [markdown]
# In the original stimulus, both target regions have the same median intensity.
# In the output array from the ODOG model, however,
# the two target regions now have a different median predicted brightness:

# %% Extract target outputs
targets_ODOG = []
for idx in np.unique(mask.astype(int)):
    if idx > 0:
        targets_ODOG.append(np.median(output_ODOG[mask == idx]))

# Visualize
plt.bar(x=["left", "right"], height=targets_ODOG, color="k")
plt.axhline(y=0.0, linestyle="dashed", color="k")
plt.xlabel("Target region")
plt.ylabel("Brightness (ODOG; median)")
plt.show()

# %% [markdown]
# Here we get a _quantitative_ prediction from the model output.
# This quantitative prediction is in the same direction.
# as the perceived brightness effect.

# %% [markdown]
# ## Comparing (F)(L)ODOG models
# The `multyscale.models` module also implements the LODOG and FLODOG models
# (Robinson, Hammon, & de Sa, 2007),
# as `.LODOG_RHS2007` and `.FLODOG_RHS2007` respectively.
# Running these models works the same way as the base ODOG model.

# %% Initialize all three models
model_LODOG = multyscale.models.LODOG_RHS2007(stimulus.shape, visextent)
model_FLODOG = multyscale.models.FLODOG_RHS2007(stimulus.shape, visextent)

# %% Integrated runs of all three models
output_LODOG = model_LODOG.apply(stimulus)
output_FLODOG = model_FLODOG.apply(stimulus)

# %% Compare outputs
# Stimulus
plt.subplot(4, 2, 1)
plt.imshow(stimulus, cmap="gray", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(4, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    stimulus[512, 256:768],
    color="black",
)

# ODOG
plt.subplot(4, 2, 3)
plt.imshow(output_ODOG, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(4, 2, 4)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    output_ODOG[512, 256:768],
    color="black",
)

# LODOG
plt.subplot(4, 2, 5)
plt.imshow(output_LODOG, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(4, 2, 6)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    output_LODOG[512, 256:768],
    color="black",
)

# FLODOG
plt.subplot(4, 2, 7)
plt.imshow(output_FLODOG, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(4, 2, 8)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    output_FLODOG[512, 256:768],
    color="black",
)

plt.show()

# %% [markdown]
# From these cut througs,
# we see that the three models overall make similar predictions for this stimulus
# but that the brightness profile is locally quite different.

# %% [markdown]
# Again, we can extract predictions for each target region
# from each model output.

# %% Extract target outputs
targets_LODOG = []
targets_FLODOG = []
for idx in np.unique(mask.astype(int)):
    if idx > 0:
        targets_LODOG.append(np.median(output_LODOG[mask == idx]))
        targets_FLODOG.append(np.median(output_FLODOG[mask == idx]))

targets = pd.DataFrame(
    {
        "ODOG": targets_ODOG,
        "LODOG": targets_LODOG,
        "FLODOG": targets_FLODOG,
    },
    index=["Left", "Right"],
)

# Visualize
targets.plot(kind="bar")
plt.axhline(y=0.0, linestyle="dashed", color="k")
plt.xlabel("Target region")
plt.ylabel("Brightness (median)")
plt.show()

# %% [markdown]
# Here we see that all three models
# _qualitative_ predict the same direction of effect,
# but make different _quantitative predictions
# for the magnitude of the brightness difference between target regions.

# %% [markdown]
# ## Model components, and step-wise execution
# The (F)(L)ODOG models consist of several components, or steps:
#
# 1. The _multiscale spatial filering_ frontend
#   _encodes_ the input (stimulus) image
#   into a representation at multiple scales/spatial frequencies, and multiple orientations.
# 2. The normalization step regulates activity in each frequency-orientation channel,
#   by normalizing it by the energy in all frequency bands
# 3. From the normalized, multichannel representation,
#   a single 2D brightness "map" is readout, by linearly combining channels.
#   From this, target predictions can be further decoded.
#
#
# The implementations in `multyscale` have the convenience `.apply()` method
# that runs an entire model in one go,
# but also provides access to each of the steps individually.


# %% [markdown]
# ### Frontend: filterbank
# _Multiscale spatial filtering_ models are defined by their multiscale spatial filtering frontend:
# A bank of filters $\mathbf{F}$ which span a range of spatial scales $S$
# are convolved with the stimulus image.
# The general -DOG family of models uses
# *D*ifference-*o*f-*G*aussian filters,
# which act as intensity-difference detecters.
# The scale of the filter directly determines the spatial frequency selectivity.
# The -ODOG subfamily of models uses *O*riented DoG filters
# that also have one of several orientations $O$.
# Thus, filter $f_{o,s}$ is a single filter in the set $\mathbf{F}$
# with orientation $o$ and scale $s$.
#
# Since each filter is 2D, it also has an implied $x,y$ pixels.
# As a result, we can also think of the 2D ($O\times S$) set $\mathbf{F}$ of filter(outputs),
# where each filter(output) $f_{o,s}$ is an image,
# as a 4D ($O \times S \times X \times Y$) set $\mathbf{I}$ of pixel intensities.
# $$ \mathbf{I}_{O \times S \times X \times Y} \equiv \mathbf{F}_{O \times S} $$
#
# The filterbank object is stored the `model.bank` attribute of the model object.
# These, and other, filterbanks are created using the `multyscale.filterbanks`-module.
# All three models here use the same filterbank
# with 6 different orientation as 7 different spatial scales (spatial frequency selecitivity).


# %% Get parameters
print(f"{model_ODOG.bank.shape[0]} orientations, {model_ODOG.bank.shape[1]} spatial scales")

# All three models have identical filters
assert np.array_equal(model_ODOG.bank.filters, model_LODOG.bank.filters)
assert np.array_equal(model_ODOG.bank.filters, model_FLODOG.bank.filters)
assert np.array_equal(model_LODOG.bank.filters, model_FLODOG.bank.filters)

# Visualise filterbank
fig, axs = plt.subplots(*model_ODOG.bank.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(model_ODOG.bank.shape[:2]):
    axs[o, s].imshow(model_ODOG.bank.filters[o, s, ...], cmap="coolwarm", extent=visextent)

# %% [markdown]
# In this visualisation of all filters,
# the rows differ in orientation of the filter
# and columns differ in the spatial scale of the filter.


# %% [markdown]
# ### Frontend: filtering
# These filters are then convolved with the stimulus image.
#
# Filterbank-objects have an `apply(...)` method,
# which filters the input stimulus with the whole bank.
# The output is an $O \times S \times Y \times X$ tensor
# of channel responses.

# %% Apply filterbank to example stimulus
filters_output = model_ODOG.bank.apply(stimulus)

print(f"{filters_output.shape} channel responses")


# Visualise each filter output
fig, axs = plt.subplots(*filters_output.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(filters_output.shape[:2]):
    axs[o, s].imshow(filters_output[o, s], cmap="coolwarm", extent=visextent)

# %% [markdown]
# In this visualisation of filter output,
# again, rows differ in orientation of the filter
# and columns differ in the spatial scale of the filter.


# %% [markdown]
# ### Frontend: weighting filter(outputs) according to CSF
# In the -ODOG models, the filter outputs are weighted according to the CSF,
# that is, higher frequencies (smaller spatial scales),
# are weighted more strongly:
# "The seven spatial frequency filters [are weighted] across frequency
#  using a power function with a slope of 0.1"
#
# This slope is an attribute of the `ODOG_RHS2007` object,
# which then determines the `scale_weights` attribute.
# The weights can be applied through the `<model>.weight_outputs` method.

# %% Weight individual filter(outputs) according to spatial size (frequency)
print(f"Slope of weights {model_ODOG.weights_slope}, gives weights:")
print(f"{model_ODOG.scale_weights}")

weighted_outputs = model_ODOG.weight_outputs(filters_output)

# Visualise weighted filter outputs
fig, axs = plt.subplots(*weighted_outputs.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(weighted_outputs.shape[:2]):
    axs[o, s].imshow(weighted_outputs[o, s], cmap="coolwarm", extent=visextent)

# Since this weighting just scales the output,
# it does not affect this visualisation of the filter outputs.

# %% [markdown]
# ### Normalization
# Each filter output then gets normalized
# by a combination of the other filter outputs.

# %% Normalize filter outputs
normalized_outputs = model_ODOG.normalize_outputs(weighted_outputs)

# Visualise normalized filter outputs
fig, axs = plt.subplots(*normalized_outputs.shape[:2], sharex="all", sharey="all")
for o, s in np.ndindex(normalized_outputs.shape[:2]):
    axs[o, s].imshow(normalized_outputs[o, s], cmap="coolwarm", extent=visextent)

# %% [markdown]
# ### Readout
# To readout from this (normalized) multiscale representation,
# first we integrate over orientations and scales,
# resulting in a 2D $(Y \times X)$ array again

# %% Sum normalized filter outputs into single array
output_stepwise = normalized_outputs.sum((0, 1))

# Visualise output
plt.subplot(1, 2, 1)
plt.imshow(output_stepwise, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(1, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    output_stepwise[512, 256:768],
    color="black",
)
plt.show()

# %% [markdown]
# In the horizontal cut on the right, we can see that the model output
# is greater for the target patch on the left side of the stimulus,
# than for the patch on the right side
# -- also indicated in the output image by the warmer color on the left than on the right.
# This is in the same direction as the perceived brightness effect.

# %% [markdown]
# This stepwise running of the model
# is exactly what the `model.apply(...)` function does,
# thus the output is identical.

# %% Compare runs
# Stimulus
plt.subplot(3, 2, 1)
plt.imshow(stimulus, cmap="gray", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(3, 2, 2)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    stimulus[512, 256:768],
    color="black",
)

# Integrated run
plt.subplot(3, 2, 3)
plt.imshow(output_ODOG, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(3, 2, 4)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    output_ODOG[512, 256:768],
    color="black",
)

# Stepwise
plt.subplot(3, 2, 5)
plt.imshow(output_stepwise, cmap="coolwarm", extent=visextent)
plt.axhline(y=0, color="black", dashes=(1, 1))
plt.subplot(3, 2, 6)
plt.plot(
    np.linspace(visextent[2], visextent[3], stimulus.shape[1])[256:768],
    output_stepwise[512, 256:768],
    color="black",
)

# Check (assert) that the two runs are identical
assert np.array_equal(output_ODOG, output_stepwise)


# %% [markdown]
# ### Overlap and distinction between F-L-ODOG steps
# The three models-species are distinct,
# and their output is different,
# but they also share some components.
#
# All three models share the same frontend (filterbank, weighting),
# but their normalization steps are different.
# Thus, one can get the (weighted) filter output from only one model,
# and run only the distinct normalization steps from the other models
# to produce the three outputs.

# %% Apply different normalizations
normalized_ODOG = model_ODOG.normalize_outputs(weighted_outputs)
normalized_LODOG = model_LODOG.normalize_outputs(weighted_outputs)
normalized_FLODOG = model_FLODOG.normalize_outputs(weighted_outputs)

output_stepwise_ODOG = normalized_ODOG.sum((0, 1))
output_stepwise_LODOG = normalized_LODOG.sum((0, 1))
output_stepwise_FLODOG = normalized_FLODOG.sum((0, 1))

assert np.array_equal(output_ODOG, output_stepwise_ODOG)
assert np.array_equal(output_LODOG, output_stepwise_LODOG)
assert np.array_equal(output_FLODOG, output_stepwise_FLODOG)
