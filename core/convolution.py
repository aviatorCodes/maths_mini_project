import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pad_image(image, padding, mode="reflect"):
    """
    Pads an image before convolution.

    Parameters
    ----------
    image : numpy.ndarray
        Input image matrix.

    padding : int
        Number of pixels to add around the image.

    mode : str
        Padding method:
        - constant
        - edge
        - reflect
    """

    if padding == 0:
        return image

    if image.ndim == 2:

        return np.pad(
            image,
            ((padding, padding), (padding, padding)),
            mode=mode
        )

    elif image.ndim == 3:

        return np.pad(
            image,
            (
                (padding, padding),
                (padding, padding),
                (0, 0)
            ),
            mode=mode
        )

    else:
        raise ValueError("Image must be 2D or 3D")


def convolve_channel(image, kernel, padding_mode="reflect"):
    """
    Performs 2D convolution using vectorized tensor dot products.
    Eliminates Python loops for massive performance gains.
    """
    image = image.astype(float)
    kernel = kernel.astype(float)

    kernel_height, kernel_width = kernel.shape

    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode=padding_mode
    )

    flipped_kernel = np.flipud(np.fliplr(kernel))
    windows = sliding_window_view(padded, window_shape=(kernel_height, kernel_width))
    output = np.tensordot(windows, flipped_kernel, axes=((2, 3), (0, 1)))

    return output


def convolve(image, kernel, padding_mode="reflect"):
    """
    Performs convolution on either:

    - grayscale image: H x W
    - RGB image: H x W x 3
    """

    image = image.astype(float)

    if image.ndim == 2:

        return convolve_channel(
            image,
            kernel,
            padding_mode
        )

    elif image.ndim == 3:

        channels = []

        for channel in range(image.shape[2]):

            result = convolve_channel(
                image[:, :, channel],
                kernel,
                padding_mode
            )

            channels.append(result)

        return np.stack(channels, axis=2)

    else:
        raise ValueError(
            "Image must be grayscale or RGB"
        )


def clip_image(image):
    """
    Restricts pixel values to the valid [0,255] range.
    """

    return np.clip(image, 0, 255)


def normalize_image(image):
    """
    Normalizes arbitrary values to [0,255].

    Useful for filters such as edge detection
    where negative values may occur.
    """

    minimum = image.min()
    maximum = image.max()

    if maximum == minimum:
        return np.zeros_like(image)

    normalized = (
        (image - minimum)
        / (maximum - minimum)
    ) * 255

    return normalized