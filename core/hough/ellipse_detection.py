"""
ellipse_detection.py — Ellipse detection using contour fitting.

Implements ellipse detection by finding contours and fitting ellipses to them.
"""

import numpy as np
import cv2


def detect_ellipses(image: np.ndarray,
                    min_semi_major: int = 15,
                    max_semi_major: int = 150,
                    threshold_percentage: float = 0.5,
                    distance: int = 20) -> list:
    """
    Detect ellipses in an image using contour-based fitting.

    Parameters
    ----------
    image                : input image (BGR or grayscale)
    min_semi_major       : minimum semi-major axis (default 15)
    max_semi_major       : maximum semi-major axis (default 150)
    threshold_percentage : not used in contour method (kept for API compatibility)
    distance             : minimum distance between ellipse centers (default 20)

    Returns
    -------
    ellipses : list of (center_x, center_y, semi_major, semi_minor, angle) tuples
    """
    # Ensure image is BGR
    if len(image.shape) == 2:
        # Grayscale image - convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Create a binary mask where colored regions are white
        _, saturation, _ = cv2.split(hsv)
        # Threshold on saturation to identify colored regions
        _, mask = cv2.threshold(saturation, 5, 255, cv2.THRESH_BINARY)
    else:
        # If grayscale, use intensity thresholding
        _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the mask
    # Apply morphological operations to enhance contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours
    min_area = 20
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    detected_ellipses = []
    
    for contour in contours:
        # Calculate moments to find centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        # Get centroid coordinates
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Get points of the contour
        points = contour.reshape(-1, 2)
        
        # Calculate covariance matrix for points
        points = points.astype(np.float64)
        points_centered = points - np.array([cx, cy])
        
        # Skip if too few points
        if len(points_centered) < 5:
            continue
        
        # Calculate covariance matrix
        cov = np.cov(points_centered.T)
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Handle case where eigenvalues are complex or one is missing
        if np.iscomplexobj(eigenvalues):
            eigenvalues = np.real(eigenvalues)
        
        # Sort eigenvalues and corresponding eigenvectors (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate major and minor axes
        # Chi-square value for 95% confidence with 2 DOF is 5.991
        major_axis = 2 * np.sqrt(5.991 * max(eigenvalues[0], 0))
        minor_axis = 2 * np.sqrt(5.991 * max(eigenvalues[1], 0))
        
        # Find maximum distance from center to any contour point
        max_distance = 0
        for point in points:
            distance_to_point = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
            max_distance = max(max_distance, distance_to_point)
        
        # Adjust axes to match actual contour extent
        if major_axis > 0:
            scale_factor = 1.01 * max_distance / (major_axis / 2)
            major_axis *= scale_factor
            minor_axis *= scale_factor
        
        # Ensure major >= minor
        if minor_axis > major_axis:
            major_axis, minor_axis = minor_axis, major_axis
        
        # Validate dimensions
        semi_major = int(major_axis / 2)
        semi_minor = int(minor_axis / 2)
        
        if semi_major < min_semi_major or semi_major > max_semi_major:
            continue
        
        if semi_minor < 3:
            semi_minor = 3
        
        # Calculate angle of orientation (in radians)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        detected_ellipses.append((float(cx), float(cy), semi_major, semi_minor, angle))
    
    # Apply non-maximum suppression to remove duplicates
    filtered_ellipses = _apply_nms_ellipses(detected_ellipses, distance)
    
    return filtered_ellipses


def _apply_nms_ellipses(ellipses, distance_threshold):
    """Remove duplicate ellipses that are too close together."""
    if not ellipses:
        return ellipses
    
    # Sort by semi_major size (descending) for NMS priority
    ellipses = sorted(ellipses, key=lambda e: e[2], reverse=True)
    
    filtered = []
    used = [False] * len(ellipses)
    
    for i, ellipse in enumerate(ellipses):
        if used[i]:
            continue
        
        cx1, cy1, a1, b1, angle1 = ellipse
        filtered.append(ellipse)
        used[i] = True
        
        # Mark nearby ellipses as used
        for j in range(i + 1, len(ellipses)):
            if used[j]:
                continue
            cx2, cy2, a2, b2, angle2 = ellipses[j]
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if dist < distance_threshold:
                used[j] = True
    
    return filtered
