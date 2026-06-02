import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.data_structures import GroupData, InstanceData
from ..glmap import GLMap

logger = logging.getLogger(__name__)


def _compute_3d_bbox(point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate 3D bounding box of point cloud"""
    if len(point_cloud) == 0:
        return np.array([]), np.array([])

    # Calculate 8 corner points of bounding box
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)

    # Coordinates of 8 corner points
    corners = np.array(
        [
            [min_coords[0], min_coords[1], min_coords[2]],  # left bottom back
            [max_coords[0], min_coords[1], min_coords[2]],  # right bottom back
            [max_coords[0], max_coords[1], min_coords[2]],  # right top front
            [min_coords[0], max_coords[1], min_coords[2]],  # left top front
            [min_coords[0], min_coords[1], max_coords[2]],  # left bottom front
            [max_coords[0], min_coords[1], max_coords[2]],  # right bottom front
            [max_coords[0], max_coords[1], max_coords[2]],  # right top front
            [min_coords[0], max_coords[1], max_coords[2]],  # left top front
        ]
    )

    return corners, np.array([min_coords, max_coords])


def _export_instance_3d_bbox(instance_data: InstanceData) -> Dict[str, Any]:
    """Export 3D bounding box information of single instance"""
    corners, bbox_extents = _compute_3d_bbox(instance_data.point_cloud)

    return {
        "instance_id": instance_data.instance_id,
        "category": instance_data.category,
        "description": instance_data.description,
        "bbox_corners": corners.tolist() if len(corners) > 0 else [],
        "bbox_extents": bbox_extents.tolist() if len(bbox_extents) > 0 else [],
        "point_count": len(instance_data.point_cloud),
    }


def _export_group_3d_bbox(gl_map: GLMap, group_data: GroupData) -> Dict[str, Any]:
    """Export 3D bounding box information of single group"""
    # Ensure instance_ids is list type, not set type
    instance_ids_list = list(group_data.instance_ids) if group_data.instance_ids else []

    if not instance_ids_list:
        return {
            "group_id": group_data.group_id,
            "description": group_data.description,
            "bbox_corners": [],
            "bbox_extents": [],
            "point_count": 0,
            "instance_ids": [],
        }

    # Merge point clouds of all instances in group
    all_points = []
    for instance_id in instance_ids_list:
        instance_data = gl_map.instance_manager.get_instance(instance_id)
        if instance_data and instance_data.point_cloud is not None:
            all_points.extend(instance_data.point_cloud.tolist())

    if not all_points:
        return {
            "group_id": group_data.group_id,
            "description": group_data.description,
            "bbox_corners": [],
            "bbox_extents": [],
            "point_count": 0,
            "instance_ids": instance_ids_list,
        }

    all_points_array = np.array(all_points)
    corners, bbox_extents = _compute_3d_bbox(all_points_array)

    return {
        "group_id": group_data.group_id,
        "description": group_data.description,
        "bbox_corners": corners.tolist() if len(corners) > 0 else [],
        "bbox_extents": bbox_extents.tolist() if len(bbox_extents) > 0 else [],
        "point_count": len(all_points_array),
        "instance_ids": instance_ids_list,
    }


def export_gl_map_instances_3d_bbox_list(
    gl_map: GLMap,
) -> List[Dict[str, Any]]:
    """Export 3D bounding box list of all instances in map"""
    bbox_list = []
    for instance_id, instance_data in gl_map.instances.items():
        if instance_data.point_cloud is not None and len(instance_data.point_cloud) > 0:
            bbox_list.append(_export_instance_3d_bbox(instance_data))

    return bbox_list


def export_gl_map_groups_3d_bbox_list(gl_map: GLMap) -> List[Dict[str, Any]]:
    """Export 3D bounding box list of all groups in map"""
    bbox_list = []
    for group_id, group_data in gl_map.group_manager.groups.items():
        bbox_list.append(_export_group_3d_bbox(gl_map, group_data))

    return bbox_list


def export_gl_map_cloud_points_to_ply(
    gl_map: GLMap, output_path: str
) -> Dict[str, Any]:
    """Export all instance point clouds in map to PLY file and return color mapping information"""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    color_mapping = {}

    with open(output_path, "w") as f:
        # Write PLY file header
        total_points = sum(
            len(instance_data.point_cloud)
            for instance_data in gl_map.instance_datas
            if instance_data.point_cloud is not None
        )

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Generated from GLMap visualization\n")
        f.write(f"comment Total instances: {len(gl_map.instances)}\n")
        f.write(f"element vertex {total_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write point data
        for instance_id, instance_data in gl_map.instances.items():
            if instance_data.point_cloud is None or len(instance_data.point_cloud) == 0:
                continue

            # Generate random color (ensure colors are different enough)
            color = np.random.randint(50, 255, size=3)  # avoid too dark colors

            # Record color mapping
            color_mapping[instance_id] = {
                "color": color.tolist(),
                "category": instance_data.category,
                "description": instance_data.description,
                "point_count": len(instance_data.point_cloud),
            }

            # Write point cloud data
            for point in instance_data.point_cloud:
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )

    return color_mapping


def export_gl_map_3d_visualization(gl_map: GLMap, output_dir: str) -> Dict[str, str]:
    """
    Export complete visualization data of map
    """
    output_files = {}
    os.makedirs(output_dir, exist_ok=True)

    # Export instance point cloud PLY file
    ply_path = os.path.join(output_dir, "instances.ply")
    output_files["ply"] = ply_path
    color_mapping = export_gl_map_cloud_points_to_ply(gl_map, ply_path)

    color_mapping_path = os.path.join(output_dir, "color_mapping.json")
    output_files["color_mapping"] = color_mapping_path
    with open(color_mapping_path, "w", encoding="utf-8") as f:
        json.dump(color_mapping, f, ensure_ascii=False, indent=2)

    # Export instance 3D bounding box JSON
    instances_bbox_path = os.path.join(output_dir, "instances_3d_bbox.json")
    output_files["instances_bbox"] = instances_bbox_path
    instances_bbox_list = export_gl_map_instances_3d_bbox_list(gl_map)
    with open(instances_bbox_path, "w", encoding="utf-8") as f:
        json.dump(instances_bbox_list, f, ensure_ascii=False, indent=2)

    # Export group 3D bounding box JSON
    groups_bbox_path = os.path.join(output_dir, "groups_3d_bbox.json")
    output_files["groups_bbox"] = groups_bbox_path
    groups_bbox_list = export_gl_map_groups_3d_bbox_list(gl_map)
    with open(groups_bbox_path, "w", encoding="utf-8") as f:
        json.dump(groups_bbox_list, f, ensure_ascii=False, indent=2)

    logger.info(f"3D visualization data exported to directory: {output_dir}")
    return output_files


# Backward compatible alias function
def export_gl_map_instances_3d_bbox_list_legacy(gl_map: GLMap):
    """Backward compatible function, returns original bounding box corner list"""
    return [
        _compute_3d_bbox(instance_data.point_cloud)[0]
        for instance_data in gl_map.instances.values()
        if instance_data.point_cloud is not None and len(instance_data.point_cloud) > 0
    ]
