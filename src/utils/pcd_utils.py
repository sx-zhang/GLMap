from typing import Optional

import cv2
import numpy as np

from utils.geometry_utils import get_point_cloud, transform_points


def downsample_point_cloud(
    point_cloud: np.ndarray,
    horizontal_voxel_size: float = 0.025,
    vertical_voxel_size: float = 0.05,
) -> np.ndarray:
    """Voxel-downsample a point cloud."""
    if len(point_cloud) == 0:
        return point_cloud

    voxel_indices_x = np.floor(point_cloud[:, 0] / horizontal_voxel_size).astype(int)
    voxel_indices_y = np.floor(point_cloud[:, 1] / horizontal_voxel_size).astype(int)
    voxel_indices_z = np.floor(point_cloud[:, 2] / vertical_voxel_size).astype(int)
    voxel_indices = np.column_stack([voxel_indices_x, voxel_indices_y, voxel_indices_z])

    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return point_cloud[np.sort(unique_indices)]


def get_object_point_clouds(
    depth: np.ndarray,
    object_masks: list[np.ndarray],
    fx: float,
    fy: float,
    min_depth: float,
    max_depth: float,
    transform: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    downsample: bool = False,
    horizontal_voxel_size: float = 0.025,
    vertical_voxel_size: float = 0.025,
) -> list[np.ndarray]:
    """Extract point clouds for each object mask from a depth image.

    Args:
        depth: Depth map (H, W) with normalized values; combined with
            min/max_depth to recover real distances.
        object_masks: List of binary masks for detected objects.
        fx, fy: Camera focal lengths.
        min_depth, max_depth: Depth sensor range.
        transform: 4x4 camera-to-world transformation matrix.
        rgb: RGB image (H, W, 3). When provided, colors are included.
        downsample: Whether to apply voxel downsampling.
        horizontal_voxel_size: Horizontal voxel size for downsampling.
        vertical_voxel_size: Vertical voxel size for downsampling.

    Returns:
        List of point clouds. Each element is (N, 6) [x, y, z, r, g, b] when
        rgb is provided, otherwise (N, 3) [x, y, z].
    """
    height, width = depth.shape[:2]
    has_color = rgb is not None
    empty = np.empty((0, 6 if has_color else 3))
    result = []

    for mask in object_masks:
        local_cloud = _extract_local_cloud(depth, mask, min_depth, max_depth, fx, fy)
        if len(local_cloud) == 0:
            result.append(empty)
            continue

        # Depth range filtering
        valid = (local_cloud[:, 0] <= max_depth * 0.95) & (
            local_cloud[:, 0] >= min_depth * 1.05
        )
        local_cloud = local_cloud[valid]
        if len(local_cloud) == 0:
            result.append(empty)
            continue

        # Back-project to image plane for RGB lookup
        if has_color:
            z = local_cloud[:, 0]
            depth_valid = z > 1e-6
            if not np.any(depth_valid):
                result.append(empty)
                continue

            z_v = z[depth_valid]
            cam_x = -local_cloud[depth_valid, 1]
            cam_y = -local_cloud[depth_valid, 2]
            u = np.clip(cam_x * fx / z_v + width / 2, 0, width - 1).astype(int)
            v = np.clip(cam_y * fy / z_v + height / 2, 0, height - 1).astype(int)

            local_cloud = local_cloud[depth_valid]
            colors = rgb[v, u, :]

        # Transform to world coordinates
        world_cloud = transform_points(transform, local_cloud)
        pcd = (
            np.concatenate([world_cloud, colors], axis=1) if has_color else world_cloud
        )

        if downsample and len(pcd) > 0:
            pcd = downsample_point_cloud(
                pcd, horizontal_voxel_size, vertical_voxel_size
            )

        result.append(pcd)

    return result


def _extract_local_cloud(
    depth: np.ndarray,
    mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    fx: float,
    fy: float,
) -> np.ndarray:
    """Extract a local point cloud in camera coordinates from depth and mask."""
    eroded = cv2.erode((mask * 255).astype(np.uint8), None, iterations=1)
    real_depth = depth * (max_depth - min_depth) + min_depth
    return get_point_cloud(real_depth, eroded, fx, fy)
