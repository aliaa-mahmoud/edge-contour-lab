"""
line_detection.py — Hough transform for line detection from scratch.

Implements the Hough transform to detect lines in edge maps with line merging.
"""

import numpy as np


def detect_lines(edges: np.ndarray,
                 threshold_percentage: float = 0.9) -> list:
    """
    Detect lines using the Hough transform with line merging (from scratch).

    Parameters
    ----------
    edges                : binary edge map (uint8, 0/255)
    threshold_percentage : percentage of max votes (0.0-1.0, default 0.9 = 90%)

    Returns
    -------
    lines : list of ((x1, y1), (x2, y2)) tuples representing line endpoints
    """
    if edges.dtype != np.uint8:
        edges = (edges > 0).astype(np.uint8) * 255
    
    height, width = edges.shape
    diagonal = int(np.sqrt(height**2 + width**2))
    
    # Rho range: -diagonal to diagonal with step 1
    rho_range = np.arange(-diagonal, diagonal + 1)
    # Theta range: -90 to 90 degrees with step 1 degree (inclusive of 90)
    theta_range = np.arange(-90, 91) * np.pi / 180
    
    # Initialize the accumulator
    accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.int32)
    
    # Get edge points
    edge_points = np.argwhere(edges > 0)
    if len(edge_points) == 0:
        return []
    
    # Vectorized voting process
    # Precompute cos and sin for all theta values
    cos_thetas = np.cos(theta_range)  # (num_thetas,)
    sin_thetas = np.sin(theta_range)  # (num_thetas,)
    
    # Extract x and y coordinates
    x = edge_points[:, 1:2]  # (N, 1)
    y = edge_points[:, 0:1]  # (N, 1)
    
    # Compute all rho values at once: (N, num_thetas)
    rhos = x @ cos_thetas[np.newaxis, :] + y @ sin_thetas[np.newaxis, :]
    rho_indices = np.round(rhos).astype(int) + diagonal
    
    # Accumulate votes using vectorized indexing
    # Create theta indices for all points
    theta_indices = np.arange(len(theta_range))[np.newaxis, :]  # (1, num_thetas)
    theta_indices = np.tile(theta_indices, (len(edge_points), 1))  # (N, num_thetas)
    
    # Filter valid indices (within bounds)
    valid = (rho_indices >= 0) & (rho_indices < len(rho_range))
    
    # Use np.add.at for efficient batch accumulation
    rho_valid = rho_indices[valid]
    theta_valid = theta_indices[valid]
    np.add.at(accumulator, (rho_valid, theta_valid), 1)
    
    # Calculate vote threshold based on percentage of maximum votes
    max_votes = np.max(accumulator)
    if max_votes == 0:
        return []  # No lines detected
    
    vote_threshold = int(max_votes * threshold_percentage)
    
    # Find detected lines using non-maxima suppression
    detected_lines_hough = _find_lines_with_nms(
        accumulator, rho_range, theta_range, vote_threshold
    )
    
    # Merge similar lines
    merged_lines_hough = _merge_similar_lines(detected_lines_hough)
    
    # Convert Hough lines to actual line segments from edge points
    lines = _extract_line_segments_from_hough(
        merged_lines_hough, edge_points
    )
    
    return lines


def _find_lines_with_nms(accumulator, rho_range, theta_range, vote_threshold):
    """Find lines using non-maxima suppression on 8-neighborhood."""
    detected_lines = []
    
    # Non-maxima suppression and threshold application
    for rho_idx in range(1, len(rho_range) - 1):
        for theta_idx in range(1, len(theta_range) - 1):
            votes = accumulator[rho_idx, theta_idx]
            
            # Check if it's above threshold
            if votes > vote_threshold:
                is_local_max = True
                
                # Check 8-neighborhood for non-maxima suppression
                for dr in [-1, 0, 1]:
                    for dt in [-1, 0, 1]:
                        if dr == 0 and dt == 0:
                            continue
                        if accumulator[rho_idx + dr, theta_idx + dt] > votes:
                            is_local_max = False
                            break
                    if not is_local_max:
                        break
                
                if is_local_max:
                    rho = rho_range[rho_idx]
                    theta = theta_range[theta_idx]
                    detected_lines.append((rho, theta, votes))
    
    # Sort lines by vote count (descending)
    detected_lines.sort(key=lambda x: x[2], reverse=True)
    
    return detected_lines


def _merge_similar_lines(lines, rho_threshold=10, theta_threshold=np.pi/36):
    """
    Merge similar lines in Hough space.
    
    Parameters
    ----------
    lines : list of (rho, theta, votes) tuples
    rho_threshold : maximum rho difference for merging (default 10)
    theta_threshold : maximum theta difference for merging in radians (default π/36 ≈ 5°)
    """
    if not lines:
        return []
    
    merged_lines = []
    lines = sorted(lines, key=lambda x: x[2], reverse=True)  # Sort by votes
    
    used = [False] * len(lines)
    
    for i, (rho1, theta1, votes1) in enumerate(lines):
        if used[i]:
            continue
        
        used[i] = True
        similar_lines = [(rho1, theta1, votes1)]
        
        for j, (rho2, theta2, votes2) in enumerate(lines[i+1:], i+1):
            if used[j]:
                continue
            
            # Check if lines are similar
            # Lines can be parallel (same theta) or opposite (theta differs by π)
            if (abs(rho1 - rho2) < rho_threshold and 
                (abs(theta1 - theta2) < theta_threshold or 
                 abs(abs(theta1 - theta2) - np.pi) < theta_threshold)):
                used[j] = True
                similar_lines.append((rho2, theta2, votes2))
        
        # Average the parameters of similar lines, weighted by votes
        total_votes = sum(line[2] for line in similar_lines)
        avg_rho = sum(line[0] * line[2] for line in similar_lines) / total_votes
        
        # Careful with theta averaging - need to handle wraparound
        sin_avg = sum(np.sin(line[1]) * line[2] for line in similar_lines) / total_votes
        cos_avg = sum(np.cos(line[1]) * line[2] for line in similar_lines) / total_votes
        avg_theta = np.arctan2(sin_avg, cos_avg)
        
        merged_lines.append((avg_rho, avg_theta, total_votes))
    
    return merged_lines


def _extract_line_segments_from_hough(hough_lines, edge_points):
    """
    Extract line segments based on actual extent of edge points.
    For each detected line, find all edge points that belong to it,
    then use their min/max extents to define the line segment.
    """
    lines = []
    
    if len(edge_points) == 0:
        return lines
    
    for rho, theta, votes in hough_lines:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Find all edge points close to this line
        # Distance from point (x, y) to line: |x*cos(theta) + y*sin(theta) - rho|
        x = edge_points[:, 1].astype(float)
        y = edge_points[:, 0].astype(float)
        
        distances = np.abs(x * cos_t + y * sin_t - rho)
        # Use adaptive tolerance based on angle (wider tolerance for edge cases)
        tolerance = 2.5
        close_points_mask = distances < tolerance
        
        if np.sum(close_points_mask) < 2:
            continue  # Not enough points for a valid line segment
        
        close_points = edge_points[close_points_mask]
        px = close_points[:, 1].astype(float)
        py = close_points[:, 0].astype(float)
        
        # Use the extent of actual edge points along the line direction
        # Direction along the line: perpendicular to (cos_t, sin_t) is (-sin_t, cos_t)
        # Project points onto this direction to get 1D coordinates along the line
        
        # Normalize to avoid precision issues
        norm = np.sqrt(sin_t**2 + cos_t**2)
        sin_t_n = sin_t / norm
        cos_t_n = cos_t / norm
        
        # Project onto line direction (-sin_t, cos_t)
        proj_coords = -px * sin_t_n + py * cos_t_n
        
        min_proj = np.min(proj_coords)
        max_proj = np.max(proj_coords)
        
        # Compute endpoints using the line equation directly
        # For a point on the line at projection coordinate t:
        # point = (rho * cos_t, rho * sin_t) + t * (-sin_t, cos_t)
        
        # But we need a reference point on the line. Use perpendicular from origin:
        # Reference: the point on the line closest to origin
        ref_x = rho * cos_t_n
        ref_y = rho * sin_t_n
        
        # Direction along line (normalized)
        dir_x = -sin_t_n
        dir_y = cos_t_n
        
        # Endpoints
        p1_x = ref_x + min_proj * dir_x
        p1_y = ref_y + min_proj * dir_y
        p2_x = ref_x + max_proj * dir_x
        p2_y = ref_y + max_proj * dir_y
        
        lines.append(((p1_x, p1_y), (p2_x, p2_y)))
    
    return lines



