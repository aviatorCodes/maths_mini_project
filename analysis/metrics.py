import numpy as np


def _prepare_images(original, processed):
    """
    Prepare original and processed images so that they
    can be compared even when one is RGB and the other
    is grayscale.
    """

    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    # RGB -> grayscale when processed image is grayscale
    if original.ndim == 3 and processed.ndim == 2:

        original = (
            0.299 * original[:, :, 0]
            + 0.587 * original[:, :, 1]
            + 0.114 * original[:, :, 2]
        )

    # Grayscale -> RGB is not necessary for our current filters,
    # but this keeps the function safe.
    elif original.ndim == 2 and processed.ndim == 3:

        processed = (
            0.299 * processed[:, :, 0]
            + 0.587 * processed[:, :, 1]
            + 0.114 * processed[:, :, 2]
        )

    if original.shape != processed.shape:
        raise ValueError(
            f"Images must have compatible dimensions. "
            f"Got {original.shape} and {processed.shape}."
        )

    return original, processed


def calculate_mse(original, processed):
    """
    Calculate Mean Squared Error (MSE).
    """

    original, processed = _prepare_images(
        original,
        processed
    )

    mse = np.mean(
        (original - processed) ** 2
    )

    return float(mse)


def calculate_psnr(original, processed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Assumes 8-bit image pixels with maximum value 255.
    """

    mse = calculate_mse(
        original,
        processed
    )

    if mse == 0:
        return float("inf")

    max_pixel = 255.0

    psnr = 10 * np.log10(
        (max_pixel ** 2) / mse
    )

    return float(psnr)
