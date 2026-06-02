import logging

logger = logging.getLogger(__name__)

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from utils.coordinate_transforms import xRyUzB_position_to_xFyLzU_position
from utils.habitat_utils import get_habitat_object_point_clouds
from vlm.api.get_object_utils import detect_objects

from .core.data_structures import CameraView
from .glmap import GLMap


def update_gl_map(
    gl_map: GLMap,
    habitat_observations: Dict,
    main_config: DictConfig,
    output_dir: Optional[Path] = None,
) -> Tuple[List[int], List[int]]:

    new_instance_ids: List[int] = []
    new_group_ids: List[int] = []

    # Get observations
    image_np = habitat_observations["rgb"]
    depth_np = habitat_observations["depth"]
    habitat_gps = habitat_observations["gps"]  # xRyUzB format, X Right, Y Up, Z Forward
    compass = float(habitat_observations["compass"])

    # Update agent position and forward vector
    agent_position = xRyUzB_position_to_xFyLzU_position(habitat_gps)
    camera_position = agent_position + np.array(
        [0.0, 0.0, 0.88]
    )  # 0.88 is the height of the camera
    gl_map.update_last_agent_coord(xFyLzU_position=agent_position)
    forward_vector = np.array([np.cos(compass), np.sin(compass), 0.0])
    gl_map.update_last_agent_ahead_coord(
        xFyLzU_position=agent_position + forward_vector
    )

    # MLLM grounding
    image = Image.fromarray(image_np)
    mllm_result = gl_map.mllm.grounding(image=image)

    if not mllm_result or "objects" not in mllm_result:
        logger.warning("MLLM returned empty result.")
        return new_instance_ids, new_group_ids

    if output_dir:
        output_path = output_dir / "mllm_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mllm_result, f, ensure_ascii=False, indent=2)

    # detect objects and segment them
    categories = []
    cat2local_id = {}
    cat2desc = {}
    for obj in mllm_result["objects"]:
        if any(k not in obj for k in ("category", "description", "instance_id")):
            continue
        cat = obj["category"]
        if cat in cat2local_id:  # dedup
            continue
        categories.append(cat)
        cat2local_id[cat] = obj["instance_id"]
        cat2desc[cat] = obj["description"]

    if not categories:
        logger.warning("No valid categories from MLLM.")
        return new_instance_ids, new_group_ids

    detector_cfg = main_config.detector
    segmented_img, scores, masks, category_indices, _ = detect_objects(
        img=image_np, object_classes=categories, cfg=detector_cfg
    )
    if output_dir:
        segmented_img = Image.fromarray(segmented_img)
        segmented_img.save(output_dir / "segmented_img.png")

    # Filter out empty masks
    non_empty = [i for i, m in enumerate(masks) if m.sum() > 0]
    if not non_empty:
        logger.warning("No valid objects detected")
        return new_instance_ids, new_group_ids

    category_indices = [category_indices[i] for i in non_empty]
    scores = [scores[i] for i in non_empty]
    masks = [masks[i] for i in non_empty]

    # Keep best scoring mask per category
    best: dict[int, int] = {}  # cat_idx -> idx
    for idx, cat_idx in enumerate(category_indices):
        if cat_idx not in best or scores[idx] > scores[best[cat_idx]]:
            best[cat_idx] = idx
    keep = sorted(best.values())
    category_indices = [category_indices[i] for i in keep]
    scores = [scores[i] for i in keep]
    masks = [masks[i] for i in keep]

    # Point cloud extraction & empty filtering
    colored_pcd_list = get_habitat_object_point_clouds(
        cfg=main_config,
        observations=habitat_observations,
        object_masks=masks,
        with_color=True,
    )
    valid = [i for i, pcd in enumerate(colored_pcd_list) if len(pcd) > 0]
    if not valid:
        return new_instance_ids, new_group_ids
    category_indices = [category_indices[i] for i in valid]
    scores = [scores[i] for i in valid]
    colored_pcd_list = [colored_pcd_list[i] for i in valid]

    # Gaussian parameter estimation
    # Add instances & estimate gaussian parameters
    local2global = {}
    for idx, cat_idx in enumerate(category_indices):
        cat = categories[cat_idx]
        positions = colored_pcd_list[idx][:, :3]
        colors_normalied = colored_pcd_list[idx][:, 3:].astype(np.float32) / 255.0

        gaussian_data = gl_map.gaussian_estimator.estimate(
            positions=positions,
            rgb_colors=colors_normalied,
        )

        weight = scores[idx] * len(positions)
        camera_view = CameraView(position=camera_position, weight=weight)

        gid = gl_map.add_instance_from_property(
            category=cat,
            description=cat2desc[cat],
            point_cloud=positions,
            gaussian_data=gaussian_data,
            camera_view=camera_view,
        )

        local_id = cat2local_id[cat]
        local2global[local_id] = gid
        new_instance_ids.append(gid)

    # Add groups
    for group in mllm_result.get("groups", []):
        if "instances" not in group or "description" not in group:
            continue
        g_instances = {
            local2global[lid] for lid in group["instances"] if lid in local2global
        }
        if len(g_instances) <= 1:
            continue
        g_id = gl_map.add_group_from_property(
            description=group["description"], instance_ids=g_instances
        )
        new_group_ids.append(g_id)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return new_instance_ids, new_group_ids
