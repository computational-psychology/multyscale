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
def ourconv(image, filt, pad=0.5):
    # pad
    padded_size = np.array(image.shape) + np.array(filt.shape)
    pad_img = pad_RHS(image, padded_size, padval=pad)
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


def weight(filter_responses):
    weighted_responses = np.ndarray(filter_responses.shape)

    # loop over the orientations
    for o in range(filter_responses.shape[0]):
        # loop over spatial frequencies
        for f in range(filter_responses.shape[1]):
            weighted_responses[o, f] = filter_responses[o, f] * w_val[f]
    return weighted_responses


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


def lodog_RMS(this_norm, sig1, sr):
    # square
    img_sqr = this_norm**2

    # create Gaussian mask
    mask = lodog_mask(sig1, sr)

    # filter the image (using unit-sum mask --> mean)
    filter_out = ourconv(img_sqr, mask, pad=0)

    # make sure there are no negative numbers due to fft inaccuracies.
    filter_out = filter_out + 1e-6

    # take the square root, last part of doing RMS
    filter_out = np.sqrt(filter_out)

    filter_out += 1e-6
    return filter_out


def lodog_mask(sig1, sr=1, o=0):
    # sig1= size of gaussian window in the direction of the filter
    # sig2= size of gaussian window perpendicular to filter
    sig2 = sig1 * sr

    # directed along main axis of filter
    rot = orientations[o] * np.pi / 180

    # create a unit volume gaussian for filtering
    mask = d2gauss(model_x, sig1, model_y, sig2, rot)
    mask = mask / mask.sum()
    return mask


# %%
def lodog_normalize(filter_responses, sig1, sr=1):
    # normalizers
    norms = lodog_normalizers(filter_responses)

    # lRMS
    RMSs = lodog_RMSs(norms, sig1, sr)

    # loop over the orientations
    normed_resps = np.ndarray(filter_responses.shape)
    for o in range(filter_responses.shape[0]):
        # loop over spatial frequencies
        for f in range(filter_responses.shape[1]):
            filter_out = RMSs[o, f]
            normed_resps[o, f] = filter_responses[o, f] / filter_out
    return normed_resps


def lodog_normalizers(filter_responses):
    norms = np.zeros(filter_responses.shape)

    # loop over the orientations
    for o in range(filter_responses.shape[0]):
        this_norm = np.zeros(filter_responses.shape[-2:])

        # loop over spatial frequencies to accumulate
        for f in range(filter_responses.shape[1]):
            this_norm += filter_responses[o, f]  # * w_val[f]

        for f in range(filter_responses.shape[1]):
            norms[o, f] = this_norm
    return norms


def lodog_RMSs(norms, sig1, sr=1):
    RMSs = np.ndarray(norms.shape)
    # loop over the orientations
    for o in range(norms.shape[0]):
        # loop over spatial frequencies to accumulate
        for f in range(norms.shape[1]):
            RMSs[o, f] = lodog_RMS(norms[o, f], sig1, sr)
    return RMSs


def flodog_normalize(filter_response, sigx, sr, sdmix):
    pass
