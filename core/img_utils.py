import numpy as np
from PIL import Image


def load_image(uploaded_file):
    """
    Loads an image using Pillow and converts it to RGB.
    """

    image = Image.open(uploaded_file).convert("RGB")

    return np.array(image)


def image_to_grayscale(image):
    """
    Converts RGB image matrix into a grayscale matrix.

    Standard weighted conversion:

    Gray = 0.299R + 0.587G + 0.114B
    """

    if image.ndim == 2:
        return image.astype(float)

    if image.shape[2] != 3:
        raise ValueError("Expected RGB image")

    grayscale = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    return grayscale


def matrix_to_image(matrix):
    """
    Converts a matrix back into a PIL image.
    """

    matrix = np.clip(matrix, 0, 255)

    matrix = matrix.astype(np.uint8)

    if matrix.ndim == 2:
        return Image.fromarray(matrix, mode="L")

    return Image.fromarray(matrix, mode="RGB")


def image_to_matrix(image):
    """
    Returns image as a floating-point matrix.
    """

    return np.array(image).astype(float)


def get_image_dimensions(image):
    """
    Returns image dimensions.
    """

    if image.ndim == 2:

        height, width = image.shape

        return {
            "height": height,
            "width": width,
            "channels": 1
        }

    height, width, channels = image.shape

    return {
        "height": height,
        "width": width,
        "channels": channels
    }