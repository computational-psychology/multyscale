# %%
import numpy as np
import matplotlib.pyplot as plt
import filters

# %% Parameters of image
shape = (1024, 1024)  # filtershape in pixels
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
plt.imshow(img, cmap="gray")

# Plot horizontal meridian
plt.subplot(1, 2, 2)
plt.plot(x[int(img.shape[0]/2)], img[int(img.shape[0]/2), ...])

# %% Elliptical gaussian (3:1 axes; rotated)
img = filters.gaussian2d(x, y, (6, 2), orientation=15)

# Plot
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")

# Plot horizontal meridian
plt.subplot(1, 2, 2)
plt.plot(x[int(img.shape[0]/2)], img[int(img.shape[0]/2), ...])
