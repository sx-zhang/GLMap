import cv2
import numpy as np


def get_point_cloud(
    depth_image: np.ndarray, mask: np.ndarray, fx: float, fy: float
) -> np.ndarray:
    """
    Convert depth image and mask to 3D point cloud

    Args:
        depth_image: Depth values in camera frame
        mask: Binary mask indicating valid pixels
        fx, fy: Camera intrinsic parameters (focal lengths)

    Returns:
        np.ndarray: 3D points in camera coordinate system [z, -x, -y]
    """
    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] // 2) * z / fx
    y = (v - depth_image.shape[0] // 2) * z / fy
    cloud = np.stack((z, -x, -y), axis=-1)

    return cloud


def transform_points(
    transformation_matrix: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Apply 4x4 transformation matrix to 3D points

    Args:
        transformation_matrix: 4x4 homogeneous transformation matrix
        points: Nx3 array of 3D points

    Returns:
        np.ndarray: Transformed 3D points
    """
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T
    return transformed_points[:, :3] / transformed_points[:, 3:]


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """
    Convert position and yaw to 4x4 transformation matrix

    Args:
        xyz: 3D position [x, y, z]
        yaw: Rotation around z-axis in radians

    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def too_close_to_left_right_edges(
    mask: np.ndarray, edge_ratio: float = 0.05, n_div: int = 4
) -> bool:
    """
    Check if binary mask is too close to left or right edge of image.
    """
    if n_div < 2:
        raise ValueError("n_div must be at least 2")

    x, y, w, h = cv2.boundingRect(mask)
    img_w = mask.shape[1]
    unit = img_w // n_div

    # Leftmost portion
    if x + w <= unit:  # Fully within left 1/n_div
        return x <= int(edge_ratio * img_w)

    # Rightmost portion
    if x >= (n_div - 1) * unit:  # Fully within right 1/n_div
        return x + w >= int((1.0 - edge_ratio) * img_w)

    # Middle region
    return False


def too_close_to_bottom_edge(
    mask: np.ndarray, bottom_ratio: float = 0.05, n_div: int = 4
) -> bool:
    """
    Check if binary mask is too close to bottom edge of image.
    """
    if n_div < 2:
        raise ValueError("n_div must be at least 2")

    _, y, _, h = cv2.boundingRect(mask)
    img_h = mask.shape[0]
    unit = img_h // n_div  # Height of each portion

    # Starting y-coordinate of bottommost portion
    last_region_start = (n_div - 1) * unit

    # Only check if mask is fully within bottommost portion
    if y >= last_region_start:
        return y + h >= int((1.0 - bottom_ratio) * img_h)
    return False
