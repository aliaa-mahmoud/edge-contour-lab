from core.canny.gradients import compute_gradients
import cv2

def compute_image_energy(image):

    # convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gradient_magnitude = compute_gradients(gray)[0]
    gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (5,5), 0) # blur edges

    # image energy
    image_energy = - gradient_magnitude # (-) to minimise energy, we want to maximise gradient magnitude (edges)

    return image_energy



def compute_internal_energy(contour, i, candidate_point, alpha=0.1, beta=0.1):

    prev_point = contour[i-1]
    next_point = contour[(i+1) % len(contour)]

    # Elasticity (distance between points)
    elastic = (candidate_point[0] - prev_point[0])**2 + (candidate_point[1] - prev_point[1])**2

    # Curvature (smoothness of the contour)
    curvature = (prev_point[0] - 2*candidate_point[0] + next_point[0])**2 + \
                (prev_point[1] - 2*candidate_point[1] + next_point[1])**2

    energy = alpha * elastic + beta * curvature

    return energy


def greedy_snake(contour, image_energy, alpha=0.1, beta=0.1):
    """
    Returns:
        Updated contour with points moved to local minimum energy positions
    """
    new_contour = contour.copy()
    h, w = image_energy.shape

    for i in range(len(contour)):
        x, y = new_contour[i]  # Use new_contour to evolve with updated neighbors
        best_energy = float("inf")
        best_point = (x, y)

        # Search in 3x3 neighbourhood for minimum energy point
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                
                # Boundary check: clamp coordinates
                nx = max(0, min(nx, w - 1))
                ny = max(0, min(ny, h - 1))
                candidate = (nx, ny)

                # Internal energy (elasticity + curvature)
                E_int = compute_internal_energy(new_contour, i, candidate, alpha, beta)
                # Image energy (attraction to edges)
                E_img = image_energy[ny, nx]
                # Total energy
                E = E_int + E_img

                if E < best_energy:
                    best_energy = E
                    best_point = candidate

        new_contour[i] = best_point

    return new_contour

