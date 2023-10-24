# %%
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import local module
import multyscale

# %% Load example stimulus
stimulus = np.asarray(Image.open("example_stimulus.png").convert("L"))
stimulus = (stimulus - stimulus.min()) / (stimulus.max() - stimulus.min())

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create model
model = multyscale.models.FLODOG_RHS2007(shape, visextent)

# %% Weighted filter outputs
filter_outputs = model.bank.apply(stimulus)
weighted_outputs = model.weight_outputs(filter_outputs)

# %% Parameterized normalization
model.spatial_window_scalar = 4
model.sdmix = 0.5
model.scale_norm_weights = multyscale.normalization.scale_norm_weights_gaussian(
    len(model.scale_weights), model.sdmix
)
model.normalization_weights = multyscale.normalization.create_normalization_weights(
    6, 7, model.scale_norm_weights, model.orientation_norm_weights
)
model.window_sigmas = np.broadcast_to(np.array(model.center_sigmas)[None, ..., None], (6, 7, 2))

# %%
output_4_05 = model.normalize_outputs(weighted_outputs).sum((0, 1))

# %%
plt.subplot(2, 1, 1)
plt.imshow(output_4_05, extent=visextent)
plt.subplot(2, 1, 2)
plt.plot(output_4_05[512, :])

# %%
