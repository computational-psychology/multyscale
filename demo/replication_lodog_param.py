# %%
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from PIL import Image

# Import local module
from multyscale import models

# %% Load example stimulus
stimulus = np.asarray(Image.open("example_stimulus.png").convert("L"))

# %% Parameters of image
shape = stimulus.shape  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = (-16, 16, -16, 16)

# %% Create models
ODOG = models.ODOG_RHS2007(shape, visextent)
LODOG = models.LODOG_RHS2007(shape, visextent)

# %%
output_odog = ODOG.apply(stimulus)
filters_output = LODOG.bank.apply(stimulus)
weighted_outputs = LODOG.weight_outputs(filters_output)
output_odog2 = ODOG.normalize_outputs(weighted_outputs).sum((0, 1))


# %%
plt.plot(output_odog[512, :] / 32, label="ODOG")
np.allclose(output_odog2, output_odog)

# %%
LODOG.window_sigma = 1
LODOG.window_sigmas = np.ones(shape=(6, 7, 2)) * LODOG.window_sigma
output_lodog1 = LODOG.normalize_outputs(weighted_outputs).sum((0, 1))

LODOG.window_sigma = 2
LODOG.window_sigmas = np.ones(shape=(6, 7, 2)) * LODOG.window_sigma
output_lodog2 = LODOG.normalize_outputs(weighted_outputs).sum((0, 1))

LODOG.window_sigma = 4
LODOG.window_sigmas = np.ones(shape=(6, 7, 2)) * LODOG.window_sigma
output_lodog4 = LODOG.normalize_outputs(weighted_outputs).sum((0, 1))

# %%
scale = 32
linestyle_cycler = cycler("linestyle", ["-", ":", "-.", "--"])
f = plt.figure(figsize=(5, 5))
plt.rc("axes", prop_cycle=linestyle_cycler)
plt.plot(output_odog[512, :] / scale, label="ODOG")
plt.plot(output_lodog4[512, :] / scale * 2, label="LODOG 4deg")
plt.plot(output_lodog2[512, :] / scale * 2, label="LODOG 2deg")
plt.plot(output_lodog1[512, :] / scale * 4.5, label="LODOG 1deg")
plt.ylim(-9, 9)
plt.yticks(np.arange(-9, 9, step=3))
plt.grid(True, axis="y")
plt.legend()
plt.savefig("RHS_Fig_3_LODOG_params.pdf", bbox_inches=0, transparent=False)
# %%
