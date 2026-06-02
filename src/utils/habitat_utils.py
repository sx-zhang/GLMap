import math

import numpy as np

from utils.coordinate_transforms import xRyUzB_position_to_xFyLzU_position
from utils.geometry_utils import xyz_yaw_to_tf_matrix
from utils.pcd_utils import get_object_point_clouds


def extract_camera_params(cfg, observations) -> dict:
    """Extract camera intrinsic/extrinsic parameters from Habitat config and observations."""
    habitat_gps = observations["gps"]
    camera_yaw = observations["compass"][0].item()
    sensor = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor

    camera_height = sensor.position[1]
    min_depth = sensor.min_depth
    max_depth = sensor.max_depth
    hfov = sensor["hfov"]
    h, w = sensor["height"], sensor["width"]

    fx = w / (2 * math.tan(hfov * np.pi / 360.0))
    fy = h / (2 * math.tan(hfov / w * h * np.pi / 360.0))

    agent_position = xRyUzB_position_to_xFyLzU_position(habitat_gps)
    camera_position = agent_position + np.array([0, 0, camera_height])
    transform = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

    return dict(
        depth=observations["depth"][:, :, 0],
        fx=fx,
        fy=fy,
        min_depth=min_depth,
        max_depth=max_depth,
        transform=transform,
    )


def get_habitat_object_point_clouds(
    cfg,
    observations,
    object_masks,
    with_color: bool = True,
    downsample: bool = False,
    horizontal_voxel_size: float = 0.025,
    vertical_voxel_size: float = 0.025,
) -> list[np.ndarray]:
    """Extract object point clouds from Habitat sensor data.

    Args:
        cfg: Habitat configuration object.
        observations: Habitat observation dict (depth, rgb, gps, compass).
        object_masks: List of binary masks for detected objects.
        with_color: Whether to include RGB colors.
        downsample: Whether to apply voxel downsampling.
        horizontal_voxel_size: Horizontal voxel size for downsampling.
        vertical_voxel_size: Vertical voxel size for downsampling.

    Returns:
        List of point clouds. Each element is (N, 6) [x, y, z, r, g, b] if
        with_color=True, otherwise (N, 3) [x, y, z].
    """
    params = extract_camera_params(cfg, observations)
    return get_object_point_clouds(
        depth=params["depth"],
        object_masks=object_masks,
        fx=params["fx"],
        fy=params["fy"],
        min_depth=params["min_depth"],
        max_depth=params["max_depth"],
        transform=params["transform"],
        rgb=observations["rgb"] if with_color else None,
        downsample=downsample,
        horizontal_voxel_size=horizontal_voxel_size,
        vertical_voxel_size=vertical_voxel_size,
    )
