# %%
# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import local module
from multyscale import filters

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

# %% Circular gaussian
img = filters.gaussian2d(x, y, (2, 2))

# Plot
plt.subplot(1, 2, 1)
plt.imshow(img, extent=visextent)

# Plot horizontal meridian
plt.subplot(1, 2, 2)
plt.plot(x[int(img.shape[0] / 2)], img[int(img.shape[0] / 2), ...])

# %% Elliptical gaussian (3:1 axes; rotated)
img = filters.gaussian2d(x, y, (6, 2), orientation=15)

# Plot
plt.subplot(1, 2, 1)
plt.imshow(img, extent=visextent)

# Plot horizontal meridian
plt.subplot(1, 2, 2)
plt.plot(x[int(img.shape[0] / 2)], img[int(img.shape[0] / 2), ...])

# %% ODOG
sigmas = ((2, 2), (2, 4))  # surround Gaussian is 2:1 in one axis

img = filters.odog(x, y, sigmas, orientation=(80, 80))

# % Apply filter
filt_img = filters.apply(stimulus, img)

# Plot stimulus
plt.subplot(1, 3, 1)
plt.imshow(stimulus, cmap="gray", extent=visextent)

# Plot filter + horizontal meridian
plt.subplot(2, 3, 2)
plt.imshow(img, extent=visextent)
plt.subplot(2, 3, 5)
plt.plot(x[int(img.shape[0] / 2)], img[int(img.shape[0] / 2), ...])

# Plot filtered image
plt.subplot(1, 3, 3)
plt.imshow(filt_img, extent=visextent)

# %%
