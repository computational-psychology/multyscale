import numpy as np
import pytest

from multyscale import filters, normalization

shape = (1024, 1024)
visextent = (-16, 16, -16, 16)
O, S = (6, 7)


def test_scale_norm_weights_equal():
    norm_weights = normalization.scale_norm_weights_equal(S)

    assert np.all(norm_weights == np.ones((S, S)))


@pytest.mark.xfail
def test_scale_norm_weights_gaussian():
    raise NotImplementedError()


def test_orientation_norm_weights():
    norm_weights = normalization.orientation_norm_weights(O)

    assert np.all(norm_weights == np.identity(O))


def test_norm_weights_combine():
    orientation_norm_weights = normalization.orientation_norm_weights(O)
    scale_norm_weights = normalization.scale_norm_weights_equal(S)
    norm_weights = normalization.create_normalization_weights(
        O,
        S,
        scale_norm_weights,
        orientation_norm_weights,
    )

    ground_truth = np.zeros((O, S, O, S))
    for o_prime, s_prime in np.ndindex(ground_truth.shape[:2]):
        ground_truth[o_prime, s_prime, o_prime, :] = np.ones((1, S))

    assert np.all(norm_weights == ground_truth)


@pytest.mark.xfail
def test_norm_coeffs():
    raise NotImplementedError()


def test_spatial_avg_kernel():
    ODOG_kernel = normalization.spatial_kernel_globalmean(shape)

    img = np.random.rand(shape[0], shape[1])

    filtered = filters.apply(img, ODOG_kernel, padval=0)

    assert np.allclose(filtered, img.mean())


def test_norm_energy():
    filters_output = np.random.rand(O, S, shape[0], shape[1])

    scale_norm_weights = normalization.scale_norm_weights_equal(S)
    orientation_norm_weights = normalization.orientation_norm_weights(O)
    normalization_weights = normalization.create_normalization_weights(
        O, S, scale_norm_weights, orientation_norm_weights
    )

    norm_coeffs = normalization.norm_coeffs(filters_output, normalization_weights)

    energies = np.ndarray(filters_output.shape)
    kernel = normalization.spatial_kernel_globalmean(shape)

    for o, s in np.ndindex(filters_output.shape[:2]):
        energies[o, s, ...] = normalization.norm_energy(norm_coeffs[o, s, ...], kernel)

    ground_truth = np.sqrt((norm_coeffs**2).mean(axis=(2, 3)) + 1e-6)
    ground_truth = np.tile(np.expand_dims(ground_truth, [2, 3]), (1, 1, shape[0], shape[1]))

    assert np.allclose(energies, ground_truth)
