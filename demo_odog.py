# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import utils
import models

# %% Load example stimulus
stimulus = np.asarray(Image.open('example_stimulus.png').convert('L'))

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create model
model = models.ODOG_BM1999(shape, visextent)

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

# %% Sum over spatial scales
spatial_scales = utils.octave_intervals(7) * 3
weights_slope = .1
scale_weights = (1 / spatial_scales) ** weights_slope

multiscale_output = np.tensordot(filters_output, scale_weights, (1, 0))

# %% Visualise oriented multiscale output
for i in range(multiscale_output.shape[0]):
    plt.subplot(multiscale_output.shape[0],
                1, i+1)
    plt.imshow(multiscale_output[i, ...], extent=visextent)

# %%  Normalize oriented multiscale outputs by their RMS
normalized_multiscale_output = np.empty(multiscale_output.shape)
for i in range(multiscale_output.shape[0]):
    image = multiscale_output[i]
    rms = np.sqrt(np.square(image).mean((-1, -2)))  # image-wide RMS
    normalized_multiscale_output[i] = image / rms

# %% Visualise normalized multiscale output
for i in range(normalized_multiscale_output.shape[0]):
    plt.subplot(normalized_multiscale_output.shape[0],
                1, i+1)
    plt.imshow(normalized_multiscale_output[i, ...], extent=visextent)

# %% Sum over orientations
output = normalized_multiscale_output.sum(0)

# %% Visualise output
plt.subplot(1, 2, 1)
plt.imshow(output, extent=visextent)
plt.subplot(1, 2, 2)
plt.plot(output[512, 250:750])
