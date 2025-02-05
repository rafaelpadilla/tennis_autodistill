import cv2
import numpy as np
from typing import Optional

def transform_image(
    image: np.ndarray,
    homography_matrix: np.ndarray,
    output_size: Optional[tuple] = None,
) -> np.ndarray:
    """
    Transform an image using the given homography matrix.

    Args:
        image (np.ndarray): The input image to be transformed.
        homography_matrix (np.ndarray): The homography matrix used for the transformation.
        output_size (tuple): The size of the output image (width, height). If output_size is not provided, use the input image size.

    Returns:
        transformed_image (np.ndarray): The transformed image.
    """
    if output_size is None:
        output_size = (image.shape[1], image.shape[0])
    # Perform the perspective transformation using the homography matrix
    transformed_image = cv2.warpPerspective(image, homography_matrix, output_size)
    return transformed_image



def transform_point(H, point):
    """
    Applies a homography matrix to a point (x, y) and returns the transformed coordinates.

    Parameters:
        H (np.ndarray): The 3x3 homography matrix.
        point (tuple): The point (x, y) to be transformed.

    Returns:
        tuple: The transformed (x', y') coordinates.
    """
    x, y = point
    # Create the homogeneous coordinate for the point (x, y)
    point_homogeneous = np.array([x, y, 1])

    # Apply the homography matrix
    transformed_point_homogeneous = H @ point_homogeneous

    # Normalize to get the 2D coordinates (x', y')
    x_prime = transformed_point_homogeneous[0] / transformed_point_homogeneous[2]
    y_prime = transformed_point_homogeneous[1] / transformed_point_homogeneous[2]

    return (x_prime.item(), y_prime.item())