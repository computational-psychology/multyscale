# %%
# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import local module
from multyscale import utils, filterbank

# %% Load example stimulus
stimulus = np.asarray(Image.open("example_stimulus.png").convert("L"))

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create image coordinate system:
axish = np.linspace(visextent[0], visextent[1], shape[0])
axisv = np.linspace(visextent[2], visextent[3], shape[1])

(x, y) = np.meshgrid(axish, axisv)

# %% Filterbank parameters
# Parameters (BM1999)
n_orientations = 6
num_scales = 7
largest_center_sigma = 3  # in degrees
center_sigmas = utils.octave_intervals(num_scales) * largest_center_sigma
cs_ratio = 2  # center-surround ratio

# Convert to filterbank parameters
orientations = np.arange(0, 180, 180 / n_orientations)
sigmas = [((s, s), (s, cs_ratio * s)) for s in center_sigmas]

# %% Create filterbank
bank = filterbank.ODOGBank(orientations, sigmas, x, y)

# %% Visualise filterbank
for i in range(bank.filters.shape[0]):
    for j in range(bank.filters.shape[1]):
        plt.subplot(
            bank.filters.shape[0],
            bank.filters.shape[1],
            i * bank.filters.shape[0] + ((j + i) * 1) + 1,
        )
        plt.imshow(bank.filters[i, j, ...], extent=visextent)

# %% Apply filterbank
filters_output = bank.apply(stimulus)

# %% Visualise filter bank output
for i in range(filters_output.shape[0]):
    for j in range(filters_output.shape[1]):
        plt.subplot(
            filters_output.shape[0],
            filters_output.shape[1],
            i * filters_output.shape[0] + ((j + i) * 1) + 1,
        )
        plt.imshow(filters_output[i, j, ...], extent=visextent)
