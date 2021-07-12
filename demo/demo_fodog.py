# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import local module
from multyscale.models import FLODOG_RHS2007, ODOG_RHS2007
import multyscale


# %%
class FODOG(ODOG_RHS2007):
    def __init__(self, shape, visextent):
        super().__init__(shape, visextent)

        self.sdmix = 0.5  # stdev of Gaussian weights for scale mixing
        self.scale_norm_weights = multyscale.normalization.scale_norm_weights_gaussian(
            len(self.scale_weights), self.sdmix
        )
        self.normalization_weights = (
            multyscale.normalization.create_normalization_weights(
                6, 7, self.scale_norm_weights, self.orientation_norm_weights
            )
        )

    pass


# %% Load example stimulus
stimulus = np.asarray(Image.open("example_stimulus.png").convert("L"))

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create models
model_ODOG = ODOG_RHS2007(shape, visextent)
model_FODOG = FODOG(shape, visextent)
model_FLODOG = FLODOG_RHS2007(shape, visextent)

# %% Run
output_ODOG = model_ODOG.apply(stimulus)
output_FODOG = model_FODOG.apply(stimulus)
output_FLODOG = model_FLODOG.apply(stimulus)

# %%
output_ODOG / output_ODOG.sum()
output_FODOG / output_FODOG.sum()
output_FLODOG / output_FLODOG.sum()

# %% Visualise all outputs
plt.subplot(3, 2, 1)
plt.imshow(output_ODOG, extent=visextent)
plt.subplot(3, 2, 2)
plt.plot(output_ODOG[512, 250:750])

plt.subplot(3, 2, 3)
plt.imshow(output_FLODOG, extent=visextent)
plt.subplot(3, 2, 4)
plt.plot(output_FLODOG[512, 250:750])

plt.subplot(3, 2, 5)
plt.imshow(output_FODOG, extent=visextent)
plt.subplot(3, 2, 6)
plt.plot(output_FODOG[512, 250:750])
