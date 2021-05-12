# %%
import multyscale
import RHS_filters
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

# %% Create RHS filterbanks, load from Matlab
rhs_bank = RHS_filters.filterbank()
rhs_matlab = np.array(io.loadmat("odog_matlab.mat")["filters"].tolist())

# %% Visualise
for i in range(rhs_bank.shape[0]):
    plt.subplot(rhs_bank.shape[0], 2, i * 2 + 1)
    plt.imshow(rhs_bank[i, 6, ...])
    plt.subplot(rhs_bank.shape[0], 2, i * 2 + 2)
    plt.imshow(rhs_matlab[i, 6, ...])
np.allclose(rhs_matlab, rhs_bank)

# %% Parameters of image
shape = (1024, 1024)  # filtershape in pixels
# visual extent, same convention as pyplot:
visextent = np.array([-0.5, 0.5, -0.5, 0.5]) * (1023 / 32)

# %% Create image coordinate system:
axish = np.linspace(visextent[0], visextent[1], shape[0])
axisv = np.linspace(visextent[2], visextent[3], shape[1])

(x, y) = np.meshgrid(axish, axisv)

# %% Circular Gaussian
std1 = 33.9411
sigmas = ((std1 / (1024 / 32)), (std1 / (1024 / 32)))
f = multyscale.filters.gaussian2d(x, y, (sigmas[0], sigmas[1]))
f = f / f.sum()
f_2 = RHS_filters.d2gauss(shape[0], std1, shape[1], std1, 0)

plt.subplot(2, 2, 1)
plt.imshow(f)
plt.subplot(2, 2, 2)
plt.imshow(f_2)
plt.subplot(2, 2, 3)
plt.plot(f[512, :])
plt.subplot(2, 2, 4)
plt.plot(f_2[512, :])

np.allclose(f, f_2)
# %% Elliptical Gaussian
std2 = 60
orientation = 40
sigmas = ((std1 / (1024 / 32)), (std2 / (1024 / 32)))
f = multyscale.filters.gaussian2d(x, y, (sigmas[0], sigmas[1]), orientation=orientation)
f = f / f.sum()
f_2 = RHS_filters.d2gauss(shape[0], std1, shape[1], std2, orientation)

plt.subplot(2, 2, 1)
plt.imshow(f)
plt.subplot(2, 2, 2)
plt.imshow(f_2)
plt.subplot(2, 2, 3)
plt.plot(f[512, :])
plt.subplot(2, 2, 4)
plt.plot(f_2[512, :])

np.allclose(f, f_2)
# %% ODOG
orientation = 150
std1 = 60
sigmas = np.array([[std1, std1], [std1, 2 * std1]]) / (1024 / 32)
rhs_odog = RHS_filters.odog(shape[0], shape[1], std1, orientation=orientation)
multy_odog = multyscale.filters.odog(
    x, y, sigmas, orientation=(orientation, orientation)
)

plt.subplot(2, 2, 1)
plt.imshow(rhs_odog)
plt.subplot(2, 2, 2)
plt.imshow(multy_odog)
plt.subplot(2, 2, 3)
plt.plot(rhs_odog[512, :])
plt.subplot(2, 2, 4)
plt.plot(multy_odog[512, :])

np.allclose(rhs_odog, multy_odog)
# %% Filterbank
multy_bank = multyscale.filterbank.BM1999(shape, visextent)

#  Visualise filterbank
for i in range(multy_bank.filters.shape[0]):
    plt.subplot(multy_bank.filters.shape[0], 2, i * 2 + 1)
    plt.imshow(multy_bank.filters[i, 6, ...], extent=visextent)
    plt.subplot(multy_bank.filters.shape[0], 2, i * 2 + 2)
    plt.imshow(rhs_bank[i, 6, ...])

np.allclose(rhs_bank, multy_bank.filters)

# %%
# Spot check
idc_filter = (5, 6)
plt.subplot(2, 2, 1)
plt.imshow(rhs_bank[idc_filter])
plt.subplot(2, 2, 2)
plt.imshow(multy_bank.filters[idc_filter])
plt.subplot(2, 2, 3)
plt.plot(rhs_bank[idc_filter][512, :])
plt.subplot(2, 2, 4)
plt.plot(multy_bank.filters[idc_filter][512, :])

# %
np.allclose(rhs_bank[idc_filter], multy_bank.filters[idc_filter])
# %%
