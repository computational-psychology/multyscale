# %%
# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import local module
from multyscale import models, filters

# %% Load example stimulus
stimulus = np.asarray(Image.open("example_stimulus.png").convert("L"))

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create model
model = models.LODOG_RHS2007(shape, visextent)

# %% Integrated run
output_1 = model.apply(stimulus)

# %% Visualise output
plt.subplot(1, 2, 1)
plt.imshow(output_1, extent=visextent)
plt.subplot(1, 2, 2)
plt.plot(output_1[512, 250:750])

# %% Visualise filterbank
for i in range(model.bank.filters.shape[0]):
    for j in range(model.bank.filters.shape[1]):
        plt.subplot(
            model.bank.filters.shape[0],
            model.bank.filters.shape[1],
            i * model.bank.filters.shape[0] + ((j + i) * 1) + 1,
        )
        plt.imshow(model.bank.filters[i, j, ...], extent=visextent)

# %% Apply filterbank
filters_output = model.bank.apply(stimulus)

# %% Visualise filter bank output
for i in range(filters_output.shape[0]):
    for j in range(filters_output.shape[1]):
        plt.subplot(
            filters_output.shape[0],
            filters_output.shape[1],
            i * filters_output.shape[0] + ((j + i) * 1) + 1,
        )
        plt.imshow(filters_output[i, j, ...], extent=visextent)

# %% Sum over spatial scales, weighting relative to scale
multiscale_output = np.tensordot(filters_output, model.scale_weights, axes=(1, 0))

# %% Visualise oriented multiscale output
for i in range(multiscale_output.shape[0]):
    plt.subplot(multiscale_output.shape[0], 1, i + 1)
    plt.imshow(multiscale_output[i, ...], extent=visextent)

# %%  Normalize oriented multiscale outputs by local mean
# Create Gaussian window
window = filters.gaussian2d(
    model.bank.x, model.bank.y, (model.window_sigma, model.window_sigma)
)

# Normalize window to unit-sum (== spatial averaging filter)
window = window / window.sum()

# Create normalizer images
normalized_multiscale_output = np.empty(multiscale_output.shape)
normalizers = np.empty(multiscale_output.shape)
for i, image in enumerate(multiscale_output):
    # Square image
    normalizer = np.square(image)

    # Apply Gaussian window
    normalizer = filters.apply(normalizer, window)

    # Square root
    normalizer = np.sqrt(normalizer)
    normalizers[i, ...] = normalizer

    # Normalize
    normalized_multiscale_output[i, ...] = image / normalizer

# %% Visualise normalized multiscale output
for i in range(normalized_multiscale_output.shape[0]):
    plt.subplot(normalized_multiscale_output.shape[0], 3, i * 3 + 1)
    plt.imshow(multiscale_output[i, ...], extent=visextent)
    plt.subplot(normalized_multiscale_output.shape[0], 3, i * 3 + 2)
    plt.imshow(normalizers[i, ...], extent=visextent)
    plt.subplot(normalized_multiscale_output.shape[0], 3, i * 3 + 3)
    plt.imshow(normalized_multiscale_output[i, ...], extent=visextent)

# %% Sum over orientations
output_2 = normalized_multiscale_output.sum(0)

# %% Visualise both outputs
plt.subplot(2, 2, 1)
plt.imshow(output_1, extent=visextent)
plt.subplot(2, 2, 2)
plt.plot(output_1[512, 250:750])

plt.subplot(2, 2, 3)
plt.imshow(output_2, extent=visextent)
plt.subplot(2, 2, 4)
plt.plot(output_2[512, 250:750])
