"""
Gradient computation using Sobel operators.
Returns magnitude, direction (in degrees), and raw Gx/Gy components.
"""

import numpy as np


# ── Sobel kernels ──────────────────────────────────────────────────────────────
SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float64)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Minimal 2-D convolution with reflect padding."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = (padded[i:i + kh, j:j + kw] * kernel).sum()
    return output


def compute_gradients(blurred: np.ndarray):
    """
    Compute image gradients using Sobel filters.

    Parameters
    ----------
    blurred : 2-D float64 array (output of Gaussian blur)

    Returns
    -------
    magnitude  : gradient magnitude  (float64, same shape)
    direction  : gradient angle in degrees, quantised to 0/45/90/135
    gx         : raw horizontal gradient
    gy         : raw vertical gradient
    """
    gx = _convolve2d(blurred, SOBEL_X)
    gy = _convolve2d(blurred, SOBEL_Y)

    magnitude = np.hypot(gx, gy)
    magnitude = (magnitude / magnitude.max() * 255.0) if magnitude.max() > 0 else magnitude

    # Angle in degrees [0, 180)
    angle_rad = np.arctan2(gy, gx)
    angle_deg = np.degrees(angle_rad) % 180

    # Quantise to 4 directions: 0, 45, 90, 135
    direction = np.zeros_like(angle_deg, dtype=np.float64)
    direction[((angle_deg >= 0)   & (angle_deg < 22.5))  |
              ((angle_deg >= 157.5) & (angle_deg < 180))] = 0
    direction[(angle_deg >= 22.5)  & (angle_deg < 67.5)]  = 45
    direction[(angle_deg >= 67.5)  & (angle_deg < 112.5)] = 90
    direction[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 135

    return magnitude, direction, gx, gy