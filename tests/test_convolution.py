import numpy as np

from core.convolution import (
    convolve,
    clip_image
)


def test_identity_kernel():

    image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    identity = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=float)

    result = convolve(
        image,
        identity,
        padding_mode="constant"
    )

    assert np.allclose(
        result,
        image
    )


def test_constant_image_with_average_kernel():

    image = np.ones(
        (5, 5),
        dtype=float
    ) * 100

    kernel = np.ones(
        (3, 3),
        dtype=float
    ) / 9

    result = convolve(
        image,
        kernel,
        padding_mode="reflect"
    )

    assert np.allclose(
        result,
        100
    )


def test_clip():

    image = np.array([
        [-10, 50, 300]
    ])

    result = clip_image(image)

    expected = np.array([
        [0, 50, 255]
    ])

    assert np.array_equal(
        result,
        expected
    )