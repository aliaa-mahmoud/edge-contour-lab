"""
core.hough — Hough transform implementations for shape detection.

Convenience re-exports so callers can do:
    from core.hough import run_hough_line_detection, run_hough_circle_detection
"""

from .line_detection import detect_lines
from .circle_detection import detect_circles
from .ellipse_detection import detect_ellipses


def run_hough_line_detection(image,
                             kernel_size: int   = 3,
                             sigma:       float = 1.4,
                             low_ratio:   float = 0.05,
                             high_ratio:  float = 0.15,
                             threshold_percentage: float = 0.9):
    """
    Hough line detection on Canny edges.

    Parameters
    ----------
    image       : 2-D (grayscale) or 3-D (colour) uint8 numpy array
    kernel_size : Gaussian kernel size (default 3)
    sigma       : Gaussian sigma (default 1.4)
    low_ratio   : double-threshold low ratio (default 0.05)
    high_ratio  : double-threshold high ratio (default 0.15)
    threshold_percentage : percentage of max votes (0.0-1.0, default 0.9 = 90%)

    Returns
    -------
    result : dict with keys
        'original'     — original image
        'edges'        — Canny edge map
        'lines'        — detected lines as list of ((x1,y1), (x2,y2)) tuples
        'marked_image' — original image with lines drawn
    """
    from core.canny import run_canny
    
    # First run Canny to get edges
    canny_result = run_canny(image, kernel_size, sigma, low_ratio, high_ratio)
    edges = canny_result['edges']
    
    # Detect lines in the edges
    lines = detect_lines(edges, threshold_percentage)
    
    # Draw lines on original image
    marked_image = _draw_lines_on_image(image, lines)
    
    return {
        'original': image,
        'edges': edges,
        'lines': lines,
        'marked_image': marked_image,
    }


def _draw_lines_on_image(image, lines):
    """Draw detected lines on a copy of the image."""
    import numpy as np
    import cv2
    
    # Make a copy to avoid modifying original
    result = image.copy()
    if result.dtype != np.uint8:
        # Normalize if needed
        mn, mx = result.min(), result.max()
        if mx > mn:
            result = ((result - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            result = np.zeros_like(result, dtype=np.uint8)
    
    # Convert to BGR if needed for drawing
    if result.ndim == 2:
        # Grayscale: convert to BGR
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif result.ndim == 3 and result.shape[2] == 3:
        # Assume RGB (from PIL), convert to BGR for OpenCV
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    elif result.ndim == 3 and result.shape[2] == 4:
        # RGBA: convert to BGR
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    
    # Draw each line in red (BGR format: B, G, R)
    for line in lines:
        (x1, y1), (x2, y2) = line
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    # Convert back to RGB for display
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result


def run_hough_circle_detection(image,
                               kernel_size: int   = 3,
                               sigma:       float = 1.4,
                               low_ratio:   float = 0.05,
                               high_ratio:  float = 0.15,
                               min_radius:  int   = 10,
                               max_radius:  int   = 100,
                               threshold_percentage: float = 0.5,
                               distance:    int   = 15):
    """
    Hough circle detection on Canny edges.

    Parameters
    ----------
    image       : 2-D (grayscale) or 3-D (colour) uint8 numpy array
    kernel_size : Gaussian kernel size
    sigma       : Gaussian sigma
    low_ratio   : double-threshold low ratio
    high_ratio  : double-threshold high ratio
    min_radius  : minimum circle radius (default 10)
    max_radius  : maximum circle radius (default 100)
    threshold_percentage : percentage of max votes (0.0-1.0, default 0.9 = 90%)
    distance    : minimum distance between circle centers (default 15)

    Returns
    -------
    result : dict with keys
        'original'     — original image
        'edges'        — Canny edge map
        'circles'      — detected circles as list of (cx, cy, radius) tuples
        'marked_image' — original image with circles drawn
    """
    from core.canny import run_canny
    
    # First run Canny to get edges
    canny_result = run_canny(image, kernel_size, sigma, low_ratio, high_ratio)
    edges = canny_result['edges']
    
    # Detect circles in the edges
    circles = detect_circles(edges, min_radius, max_radius, threshold_percentage, distance)
    
    # Draw circles on original image
    marked_image = _draw_circles_on_image(image, circles)
    
    return {
        'original': image,
        'edges': edges,
        'circles': circles,
        'marked_image': marked_image,
    }


def run_hough_ellipse_detection(image,
                                kernel_size: int   = 3,
                                sigma:       float = 1.4,
                                low_ratio:   float = 0.05,
                                high_ratio:  float = 0.15,
                                min_semi_major: int = 15,
                                max_semi_major: int = 150,
                                threshold_percentage: float = 0.5,
                                distance:    int   = 20):
    """
    Hough ellipse detection on Canny edges.

    Parameters
    ----------
    image       : 2-D (grayscale) or 3-D (colour) uint8 numpy array
    kernel_size : Gaussian kernel size
    sigma       : Gaussian sigma
    low_ratio   : double-threshold low ratio
    high_ratio  : double-threshold high ratio
    min_semi_major : minimum semi-major axis (default 15)
    max_semi_major : maximum semi-major axis (default 150)
    threshold_percentage : percentage of max votes (0.0-1.0, default 0.9 = 90%)
    distance    : minimum distance between ellipse centers (default 20)

    Returns
    -------
    result : dict with keys
        'original'     — original image
        'edges'        — Canny edge map
        'ellipses'     — detected ellipses as list of (cx, cy, a, b, angle) tuples
        'marked_image' — original image with ellipses drawn
    """
    from core.canny import run_canny
    
    # First run Canny to get edges
    canny_result = run_canny(image, kernel_size, sigma, low_ratio, high_ratio)
    edges = canny_result['edges']
    
    # Detect ellipses in the original image (not edges)
    ellipses = detect_ellipses(image, min_semi_major, max_semi_major, threshold_percentage, distance)
    
    # Draw ellipses on original image
    marked_image = _draw_ellipses_on_image(image, ellipses)
    
    return {
        'original': image,
        'edges': edges,
        'ellipses': ellipses,
        'marked_image': marked_image,
    }


def _draw_circles_on_image(image, circles):
    """Draw detected circles on a copy of the image."""
    import numpy as np
    import cv2
    
    # Make a copy to avoid modifying original
    result = image.copy()
    if result.dtype != np.uint8:
        # Normalize if needed
        mn, mx = result.min(), result.max()
        if mx > mn:
            result = ((result - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            result = np.zeros_like(result, dtype=np.uint8)
    
    # Convert to BGR if needed for drawing
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif result.ndim == 3 and result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    elif result.ndim == 3 and result.shape[2] == 4:
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    
    # Draw each circle in green
    for cx, cy, r in circles:
        cv2.circle(result, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
    
    # Convert back to RGB for display
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result


def _draw_ellipses_on_image(image, ellipses):
    """Draw detected ellipses on a copy of the image."""
    import numpy as np
    import cv2
    
    # Make a copy to avoid modifying original
    result = image.copy()
    if result.dtype != np.uint8:
        # Normalize if needed
        mn, mx = result.min(), result.max()
        if mx > mn:
            result = ((result - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            result = np.zeros_like(result, dtype=np.uint8)
    
    # Convert to BGR if needed for drawing
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif result.ndim == 3 and result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    elif result.ndim == 3 and result.shape[2] == 4:
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    
    # Draw each ellipse in blue
    for cx, cy, a, b, angle in ellipses:
        cv2.ellipse(result, (int(cx), int(cy)), (int(a), int(b)), 
                   int(np.degrees(angle)), 0, 360, (255, 0, 0), 2)
    
    # Convert back to RGB for display
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result
