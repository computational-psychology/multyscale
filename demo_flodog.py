# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models
import filters

# %% Load example stimulus
stimulus = np.asarray(Image.open('example_stimulus.png').convert('L'))

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create model
model = models.LODOG_RHS2007(shape, visextent)

# %% Visualise filterbank
for i in range(model.bank.filters.shape[0]):
    for j in range(model.bank.filters.shape[1]):
        plt.subplot(model.bank.filters.shape[0],
                    model.bank.filters.shape[1],
                    i*model.bank.filters.shape[0]+((j+i)*1)+1)
        plt.imshow(model.bank.filters[i, j, ...], extent=visextent)

# %% Apply filterbank
filters_output = model.bank.apply(stimulus)

# %% Visualise filter bank output
for i in range(filters_output.shape[0]):
    for j in range(filters_output.shape[1]):
        plt.subplot(filters_output.shape[0],
                    filters_output.shape[1],
                    i*filters_output.shape[0]+((j+i)*1)+1)
        plt.imshow(filters_output[i, j, ...], extent=visextent)


# %% Weight each filter output according to scale
for i in range(filters_output.shape[0]):
    for j, output in enumerate(filters_output[i, ...]):
        filters_output[i, j, ...] = output * model.scale_weights[j]

# %% Build normalizer for each filter output
center_sigmas = [sigma[0][0] for sigma in model.bank.sigmas]
sdmix = .5  # stdev of Gaussian weights for scale mixing

# Create normalizer images
normalizers = np.empty(filters_output.shape)
for o, multiscale in enumerate(filters_output):  # seperate for orientations
    for i, filt in enumerate(multiscale):
        normalizer = np.empty(filt.shape)

        # Identify relative index of each scale to the current one
        rel_i = i - np.asarray(range(multiscale.shape[0]))

        # Gaussian weights, based on relative index
        gweights = np.exp(- rel_i ** 2 / 2 * sdmix ** 2) /\
            (sdmix * np.sqrt(2 * np.pi))

        # Sum filter outputs, by Gaussian weights
        normalizer = np.tensordot(multiscale, gweights, axes=(0, 0))

        # Normalize normalizer...
        area = gweights.sum()
        normalizer = normalizer / area

        # Accumulate
        normalizers[o, i, ...] = normalizer

# %% Blur normalizers
# Create Gaussian window
window = filters.gaussian2d(model.bank.x, model.bank.y,
                            (model.window_sigma, model.window_sigma))

# Normalize window to unit-sum (== spatial averaging filter)
window = window / window.sum()

for o, multiscale in enumerate(normalizers):
    for s, normalizer in enumerate(multiscale):
        # Square image
        normalizer = np.square(normalizer)

        # Apply Gaussian window
        normalizer = filters.apply(normalizer, window)

        # Square root
        normalizer = np.sqrt(normalizer)
        normalizers[o, s, ...] = normalizer

# %% Normalize filter output
normalized_outputs = np.empty(filters_output.shape)
for o, multiscale in enumerate(filters_output):
    for s, output in enumerate(multiscale):
        normalized_outputs[o, s] = output / normalizers[o, s]

# %% Sum over orientations
output_2 = normalized_outputs.sum((0, 1))

# %% Visualise both outputs
plt.subplot(2, 2, 3)
plt.imshow(output_2, extent=visextent)
plt.subplot(2, 2, 4)
plt.plot(output_2[512, 250:750])
