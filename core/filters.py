import numpy as np

from .convolution import (
    convolve,
    clip_image,
    normalize_image
)

from .kernels import (
    get_kernel,
    sobel_x,
    sobel_y
)


def apply_standard_filter(
    image,
    filter_name,
    padding_mode="reflect"
):
    """
    Applies a standard kernel-based filter.
    """

    kernel = get_kernel(filter_name)

    result = convolve(
        image,
        kernel,
        padding_mode
    )

    if filter_name in [
        "Edge Detection",
        "Emboss"
    ]:

        result = normalize_image(result)

    else:

        result = clip_image(result)

    return result


def apply_sobel(image, padding_mode="reflect"):
    """
    Applies Sobel filtering.

    First calculate:

        Gx = I * Kx
        Gy = I * Ky

    Then calculate gradient magnitude:

        G = sqrt(Gx^2 + Gy^2)
    """

    if image.ndim == 3:

        grayscale = (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        )

    else:

        grayscale = image

    gx = convolve(
        grayscale,
        sobel_x(),
        padding_mode
    )

    gy = convolve(
        grayscale,
        sobel_y(),
        padding_mode
    )

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    return normalize_image(magnitude)


def apply_standard_filter(image, filter_name, padding_mode="reflect", **kwargs):
    kernel = get_kernel(filter_name, **kwargs)
    result = convolve(image, kernel, padding_mode)

    if filter_name in ["Edge Detection", "Emboss"]:
        result = normalize_image(result)
    else:
        result = clip_image(result)
    return result

def apply_filter(image, filter_name, padding_mode="reflect", **kwargs):
    if filter_name == "Sobel":
        return apply_sobel(image, padding_mode)
    
    return apply_standard_filter(image, filter_name, padding_mode, **kwargs)