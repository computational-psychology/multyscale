import numpy as np
from scipy import fft

# %% Model params
orientations = np.linspace(0, 150, 6)  # 6 steps of 30 degrees
freqs = np.arange(0, 7)  # 7 different mechanisms

# conversion factors
DEG_PER_PIXEL = 0.03125
SPACE_CONST_TO_WIDTH = 2 * np.sqrt(np.log(2))
SPACE_CONST_TO_STD = 1 / np.sqrt(2)
STD_TO_SPACE_CONST = 1 / SPACE_CONST_TO_STD

space_const = 2**freqs * 1.5  # in pixels

# matches Table 1 in BM(1999)
space_const_deg = space_const * DEG_PER_PIXEL  # in deg.

# compute the standard deviations of the different Gaussian in pixels
# space_const = 2.^freqs * 1.5; % space constant of Gaussians
stdev_pixels = space_const * SPACE_CONST_TO_STD  # in pixels
std = space_const_deg * SPACE_CONST_TO_STD  # in degrees

# (almost matches) points along x-axis of Fig. 10 BM(1997)
cpd = 1 / (2 * space_const_deg * SPACE_CONST_TO_WIDTH)

# (almost matches) points along y-axis of Fig. 10 BM(1997)
w_val = cpd**0.1
w_val = w_val / w_val[int(np.ceil(w_val.size / 2)) - 1]
w_val = np.round(w_val, 5)

# CSF as proposed by Manos and Sakrison
# J. L. Mannos, D. J. Sakrison,
# "The Effects of a Visual Fidelity Criterion on the Encoding of Images",
# IEEE Transactions on Information Theory, pp. 525-535, Vol. 20, No 4, (1974)
CSF = 2.6 * (0.0192 + 0.114 * cpd) * np.exp(-0.114 * cpd) ** 1.1

# model size
model_y = 1024
model_x = 1024


# %%
def gauss(x, std):
    return np.exp(-(x**2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))


def d2gauss(n1, std1, n2, std2, theta):
    # rotation transformation
    theta = np.deg2rad(90 - theta)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # create X and Y grids
    Xs = np.linspace(-((n1 - 1) / 2), (n1 - 1) / 2, n1)
    Xs, Ys = np.meshgrid(Xs, Xs)

    # reshape into vectors
    Xs = Xs.T.reshape(-1)
    Ys = Ys.T.reshape(-1)
    coor = r @ np.vstack((Xs, Ys))

    # compute 1-D gaussians
    gausX = gauss(coor[0, :], std1)
    gausY = gauss(coor[1, :], std2)

    # element-wise multiplication creates 2-D gaussians
    h = np.reshape(gausX * gausY, (n2, n1))
    h = h / h.sum()

    return h


def dog(rows, columns, std1, std2, sr, theta):
    return d2gauss(rows, std1, columns, std2, theta) - d2gauss(
        rows, std1, columns, std2 * sr, theta
    )


def odog(model_x, model_y, stdev_pixels, orientation):
    return dog(model_y, model_x, stdev_pixels, stdev_pixels, 2, orientation)


# %% Filterbank
def filterbank():
    filters = np.empty((orientations.size, stdev_pixels.size, model_x, model_y))
    for i, orient in enumerate(orientations):
        for j, stdev in enumerate(stdev_pixels):
            filters[i, j, ...] = odog(model_x, model_y, stdev, orient)
    return filters


# %% Convolution
def ourconv(image, filt):
    # pad
    padded_size = np.array(image.shape) + np.array(filt.shape)
    pad_img = pad_RHS(image, padded_size, padval=0.5)
    pad_filt = pad_RHS(filt, padded_size, padval=0)

    # Paul's slightly corrected version
    temp = np.real(fft.ifft2(fft.fft2(pad_img) * fft.fft2(pad_filt)))

    # extract the appropriate portion of the filtered image
    filtered = unpad_RHS(temp, image.shape)

    return filtered


def pad_RHS(image, shape, padval):
    # pad the images
    pad_img = np.ones(shape) * padval
    pad_img[0 : image.shape[0], 0 : image.shape[1]] = image
    return pad_img


def unpad_RHS(pad_image, shape):
    image = pad_image[
        int(shape[0] / 2) : -int(shape[0] / 2),
        int(shape[1] / 2) : -int(shape[1] / 2),
    ]
    return image


# %% Normalizations
def odog_normalize(filter_responses):
    # to hold model output
    modelOut = np.zeros(filter_responses.shape[-2:])

    # loop over the orientations
    for o in range(filter_responses.shape[0]):
        this_norm = np.zeros(filter_responses.shape[-2:])
        # loop over spatial frequencies
        for f in range(filter_responses.shape[1]):
            # get the filtered response
            filt_img = filter_responses[o, f]

            # create the proper weight
            temp = filt_img * w_val[f]

            this_norm = temp + this_norm
        # do normalization
        this_norm = this_norm / np.sqrt(np.mean(this_norm * this_norm))

        # add in normalized image
        modelOut = modelOut + this_norm
    return modelOut


# %%
