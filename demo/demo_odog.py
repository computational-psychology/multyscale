# %%
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import local module
from multyscale import models

# %% Load example stimulus
stimulus = np.asarray(Image.open("example_stimulus.png").convert("L"))

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create model
model = models.ODOG_RHS2007(shape, visextent)

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

# %%  Normalize oriented multiscale outputs by their RMS
normalized_multiscale_output = np.empty(multiscale_output.shape)
rms = np.ndarray(6)
for i in range(multiscale_output.shape[0]):
    image = multiscale_output[i]
    rms[i] = np.sqrt(np.square(image).mean((-1, -2)))  # image-wide RMS
    normalized_multiscale_output[i] = image / rms[i]

# %% Visualise normalized multiscale output
for i in range(normalized_multiscale_output.shape[0]):
    plt.subplot(normalized_multiscale_output.shape[0], 1, i + 1)
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
