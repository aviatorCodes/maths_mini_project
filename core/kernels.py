import numpy as np

def box_blur(size=5):
    """
    Dynamic averaging kernel.
    A larger 'size' produces a stronger blur. Must be an odd integer.
    """
    return np.ones((size, size), dtype=float) / (size ** 2)

def gaussian_blur(size=15, sigma=3.0):
    """
    Dynamic 2D Gaussian blur.
    """
    offset = size // 2
    x, y = np.mgrid[-offset:offset+1, -offset:offset+1]
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normalizer
    return kernel / kernel.sum()

def sharpen(amount=1.0):
    """
    Dynamic sharpen kernel. 
    Higher 'amount' increases the difference between the center and neighbors.
    """
    center_weight = 4.0 + amount
    return np.array([
        [0, -1, 0],
        [-1, center_weight, -1],
        [0, -1, 0]
    ], dtype=float) / amount

def edge_detection():
    """
    Laplacian-style edge detection.
    """
    return np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=float)

def sobel_x():
    """
    Detects intensity changes in the horizontal direction.
    """
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

def sobel_y():
    """
    Detects intensity changes in the vertical direction.
    """
    return np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=float)

def emboss(intensity=1.0):
    """
    Dynamic embossing effect. 
    Higher intensity pushes the high/low values further apart.
    """
    return np.array([
        [-2 * intensity, -1 * intensity, 0],
        [-1 * intensity,  1,             1 * intensity],
        [ 0,              1 * intensity, 2 * intensity]
    ], dtype=float)

def get_kernel(name, **kwargs):
    """
    Returns the kernel corresponding to a filter name, passing any dynamic kwargs.
    """
    kernels = {
        "Box Blur": box_blur,
        "Gaussian Blur": gaussian_blur,
        "Sharpen": sharpen,
        "Edge Detection": edge_detection,
        "Sobel X": sobel_x,
        "Sobel Y": sobel_y,
        "Emboss": emboss,
    }

    if name not in kernels:
        raise ValueError(f"Unknown filter: {name}")

    return kernels[name](**kwargs)