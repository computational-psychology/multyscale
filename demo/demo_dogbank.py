# %%
# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import local module
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from multyscale import utils, filterbank

# %% Load example stimulus
stimulus = np.asarray(Image.open('example_stimulus.png').convert('L'))

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create image coordinate system:
axish = np.linspace(visextent[0], visextent[1], shape[0])
axisv = np.linspace(visextent[2], visextent[3], shape[1])

(x, y) = np.meshgrid(axish, axisv)

# %% Filterbank parameters
num_scales = 7
largest_center_sigma = 3  # in degrees
center_sigmas = utils.octave_intervals(num_scales) * largest_center_sigma
cs_ratio = 2  # center-surround ratio

# Convert to filterbank parameters
sigmas = [(s, cs_ratio*s) for s in center_sigmas]

# %% Create filterbank
bank = filterbank.DOGBank(sigmas, x, y)

# %% Visualise filterbank
for i in range(bank.filters.shape[0]):
    plt.subplot(1, bank.filters.shape[0], i+1)
    plt.imshow(bank.filters[i, ...], extent=visextent)

# %% Apply filterbank
filters_output = bank.apply(stimulus)

# %% Visualise filter bank output
for i in range(filters_output.shape[0]):
    plt.subplot(1, filters_output.shape[0], i+1)
    plt.imshow(filters_output[i, ...], extent=visextent)