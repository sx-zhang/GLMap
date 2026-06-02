import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..glmap import GLMap

logger = logging.getLogger(__name__)


def visualize_gl_map(
    gl_map: GLMap,
    output_path: Optional[Path] = None,
    show_instances: bool = True,
    show_groups: bool = True,
    show_obstacles: bool = True,
    show_agent: bool = True,
    show_path: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Visualize map, return BGR format image
    """
    height, width = gl_map.shape
    bgr_image = (
        np.ones((height, width, 3), dtype=np.uint8) * 240
    )  # light gray background

    # Draw obstacles
    if show_obstacles:
        for row, col in gl_map.obstacle_coords:
            bgr_image[row, col] = (128, 128, 128)  # gray

    # Draw instances
    classname2bgr: Dict[str, Tuple[int, int, int]] = {}
    if show_instances:
        num_classnames = len(gl_map.classnames)
        for i, classname in enumerate(gl_map.classnames):
            hue = int(180 * i / num_classnames) if num_classnames > 0 else 0
            saturation = 150  # moderate saturation
            value = 200  # higher brightness, ensure bright colors
            hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
            bgr_color = tuple(
                map(int, cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0])
            )
            classname2bgr[classname] = bgr_color

        for row, col in gl_map.valid_instances_coords:
            instances = gl_map.get_instances_at_map_coords(row, col)
            # For pixels with multiple instances, use first instance's color
            inst_classname = instances[0].category
            if inst_classname in classname2bgr:
                bgr = classname2bgr[inst_classname]
                bgr_image[row, col] = bgr

    # Draw groups
    group_ids2bgr: Dict[int, Tuple[int, int, int]] = {}
    if show_groups:
        num_groups = len(gl_map.group_manager.groups)
        for group_id in gl_map.group_manager.groups:
            hue = int(180 * group_id / num_groups) if num_groups > 0 else 0
            saturation = 150  # moderate saturation
            value = 200  # higher brightness, ensure bright colors
            hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
            bgr_color = tuple(
                map(int, cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0])
            )
            group_ids2bgr[group_id] = bgr_color

        for group_id in gl_map.group_manager.groups:
            group_data = gl_map.group_manager.get_group(group_id)
            for instance_id in group_data.instance_ids:
                map_bbox_rcrc = gl_map.instance_manager.get_instance(
                    instance_id
                ).map_bbox_rcrc
                bgr = group_ids2bgr[group_id]
                # Note: OpenCV image operations require (col, row) to correspond to (x, y)
                top_left = (map_bbox_rcrc[1], map_bbox_rcrc[0])
                bottom_right = (map_bbox_rcrc[3], map_bbox_rcrc[2])
                cv2.rectangle(bgr_image, top_left, bottom_right, bgr, thickness=2)

    if show_path:
        path_color_bgr = (0, 255, 0)
        for row, col in gl_map.agent_coord_path:
            # Note: OpenCV image operations require (col, row) to correspond to (x, y)
            cv2.circle(
                img=bgr_image,
                center=(col, row),
                radius=3,
                color=path_color_bgr,
                thickness=-1,
            )

    if show_agent:
        agent_current_map_coord = gl_map.last_agent_coord
        agent_ahead_map_coord = gl_map.last_agent_ahead_coord
        if agent_current_map_coord is not None:
            agent_radius = 5  # circle radius
            agent_color_bgr = (0, 255, 0)
            # Note: OpenCV image operations require (col, row) to correspond to (x, y)
            cv2.circle(
                img=bgr_image,
                center=(
                    agent_current_map_coord[1],
                    agent_current_map_coord[0],
                ),
                radius=agent_radius,
                color=agent_color_bgr,
                thickness=-1,
            )

            # 1.2 Draw forward orientation arrow (based on current position and forward reference point)
            if (
                agent_ahead_map_coord is not None
                and agent_ahead_map_coord != agent_current_map_coord
            ):
                # Note: OpenCV image operations require (col, row) to correspond to (x, y)
                arrow_start = (agent_current_map_coord[1], agent_current_map_coord[0])
                arrow_end = (agent_ahead_map_coord[1], agent_ahead_map_coord[0])
                arrow_color_bgr = (0, 0, 255)
                arrow_thickness = 2  # line width
                cv2.arrowedLine(
                    img=bgr_image,
                    pt1=arrow_start,
                    pt2=arrow_end,
                    color=arrow_color_bgr,
                    thickness=arrow_thickness,
                    tipLength=0.2,
                )

    if show_labels and show_instances and classname2bgr:
        bgr_image = _draw_label_bar(bgr_image, classname2bgr)

    if show_labels and show_groups and group_ids2bgr:
        group_label2bgr = {}
        for group_id, group_data in gl_map.group_manager.groups.items():
            group_label2bgr[group_data.description] = group_ids2bgr[group_id]
        bgr_image = _draw_label_bar(bgr_image, group_label2bgr)

    # Save image
    if output_path:
        if output_path.parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), bgr_image)
        logger.info(f"Saved visualization image to {output_path}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def _draw_label_bar(
    bgr_img: np.ndarray,
    label2bgr: Dict[str, Tuple[int, int, int]],
    bar_width: int = 200,
    pad: int = 10,
) -> np.ndarray:
    """
    Append vertical color bar to right side of canvas, return concatenated BGR image
    """
    h, w = bgr_img.shape[:2]
    bar = np.ones((h, bar_width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    dy = 25  # line height

    y0 = pad
    for label, bgr_color in label2bgr.items():
        cv2.rectangle(bar, (pad, y0), (pad + 20, y0 + 20), bgr_color, -1)
        cv2.putText(
            bar,
            label,
            (pad + 25, y0 + 15),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y0 += dy
        if y0 + dy > h:  # stop if exceeding height
            break

    # Concatenate
    vis = np.concatenate([bgr_img, bar], axis=1)
    return vis
