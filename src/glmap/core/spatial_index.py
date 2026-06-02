import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .data_structures import GroupData, InstanceData
from .group_manager import GroupManager
from .instance_manager import InstanceManager
from .mapping_utils import MappingUtils


class SpatialIndex:
    """Spatial index manager"""

    def __init__(
        self,
        instance_manager: InstanceManager,
        group_manager: GroupManager,
    ):
        self.logger = logging.getLogger(__name__)

        # Instance and group managers
        self.instance_manager: InstanceManager = instance_manager
        self.group_manager: GroupManager = group_manager

        # Instance level indices
        self._instance_to_indexes: Dict[int, Set[int]] = {}
        self._index_to_instances: Dict[int, Set[int]] = {}
        self._instance_bboxes: Dict[int, Tuple[int, int, int, int]] = {}

        # Group level indices
        self._index_to_groups: Dict[int, Set[int]] = {}
        self._group_to_indexes: Dict[int, Set[int]] = {}

    # ==================== get instance ====================
    @property
    def valid_instances_coords(self) -> List[Tuple[int, int]]:
        """
        Get list of pixel coordinates on map containing instances (cache preferred, lazy update)
        """
        valid_instances_indexes = list(self._index_to_instances.keys())
        valid_instances_coords = MappingUtils.index_to_coord_batch(
            valid_instances_indexes
        )
        return valid_instances_coords

    def get_instance_ids_at_episodic_position(
        self, episodic_x: float, episodic_y: float
    ) -> List[int]:
        """Get instance list at position from world coordinates in meters"""
        row, col = MappingUtils.episodic_position_to_map_coord(episodic_x, episodic_y)
        return self.get_instance_ids_at_map_coords(row, col)

    def get_instance_ids_at_map_coords(self, row: int, col: int) -> List[int]:
        """Get instance list at position from map coordinates"""
        # Check if coordinates are within valid range
        if not (
            0 <= col < MappingUtils.MAP_WIDTH and 0 <= row < MappingUtils.MAP_HEIGHT
        ):
            return []

        pixel_index = MappingUtils.coord_to_index(row, col)
        if pixel_index in self._index_to_instances:
            return list(self._index_to_instances[pixel_index].copy())
        return []

    def get_instance_ids_at_episodic_point_cloud(
        self, episodic_point_cloud: np.ndarray
    ) -> List[int]:
        """Fast bbox-based inexact search: get existing instance ID list at same position as current instance"""
        if episodic_point_cloud is None or len(episodic_point_cloud) == 0:
            return []

        # Calculate bbox of current point cloud
        querry_bbox = MappingUtils.point_cloud_to_map_bbox_rcrc(episodic_point_cloud)
        if querry_bbox is None:
            return []

        querry_min_row, querry_min_col, querry_max_row, querry_max_col = querry_bbox

        # Check if all existing instance bboxes overlap with current bbox
        overlapping_instances = set()
        for instance_id, instance_bbox in self._instance_bboxes.items():
            instace_min_row, instace_min_col, instace_max_row, instace_max_col = (
                instance_bbox
            )
            row_overlap = not (
                instace_max_row < querry_min_row or querry_max_row < instace_min_row
            )
            col_overlap = not (
                instace_max_col < querry_min_col or querry_max_col < instace_min_col
            )
            if row_overlap and col_overlap:
                overlapping_instances.add(instance_id)

        # Verify instances are still valid
        valid_instance_set = set(self.instance_manager.instance_ids)
        existing_instance_ids = overlapping_instances & valid_instance_set

        return list(existing_instance_ids)

    def get_instance_ids_at_episodic_point_cloud_exactly(
        self, episodic_point_cloud: np.ndarray
    ) -> List[int]:
        """Get existing instance ID list at same position as current instance"""
        if episodic_point_cloud is None or len(episodic_point_cloud) == 0:
            return []

        episodic_position = episodic_point_cloud[:, :2]
        map_coords = MappingUtils.episodic_position_to_map_coord_batch(
            episodic_position
        )

        valid_mask = (
            (map_coords[:, 1] >= 0)
            & (map_coords[:, 1] < MappingUtils.MAP_WIDTH)  # col valid
            & (map_coords[:, 0] >= 0)
            & (map_coords[:, 0] < MappingUtils.MAP_HEIGHT)  # row valid
        )
        valid_rows = map_coords[valid_mask, 0]  # valid pixel row array
        valid_cols = map_coords[valid_mask, 1]  # valid pixel col array

        if len(valid_rows) == 0:
            return []

        # Use sparse storage for fast query of instance IDs of all valid pixels
        temp_ids_set = set()
        for row, col in zip(valid_rows, valid_cols):
            pixel_index = MappingUtils.coord_to_index(row, col)
            if pixel_index in self._index_to_instances:
                temp_ids_set.update(self._index_to_instances[pixel_index])

        if not temp_ids_set:
            return []

        valid_instance_set = set(self.instance_manager.instance_ids)
        existing_instance_ids = temp_ids_set & valid_instance_set
        return list(existing_instance_ids)

    # ==================== modify instance ====================

    def add_instance(self, instance_data: InstanceData) -> None:
        """Add instance to region based on instance point cloud"""
        if instance_data.point_cloud is None or len(instance_data.point_cloud) == 0:
            return

        instance_id = instance_data.instance_id
        instance_point_cloud = instance_data.point_cloud
        map_mask = MappingUtils.point_cloud_to_map_mask(instance_point_cloud)
        self.add_instance_in_map_mask(instance_id, map_mask)

    def update_instance(self, instance_id: int) -> None:
        """Update instance information in spatial index based on instance point cloud"""
        instance_data: InstanceData = self.instance_manager.instances[instance_id]
        if instance_data.point_cloud is None or len(instance_data.point_cloud) == 0:
            return

        point_cloud = instance_data.point_cloud
        region_mask = MappingUtils.point_cloud_to_map_mask(point_cloud)
        self.add_instance_in_map_mask(instance_id, region_mask)

    def add_instance_in_map_mask(self, instance_id: int, map_mask: np.ndarray) -> None:
        """Add instance from region"""
        # Update instance bounding box
        self._instance_bboxes[instance_id] = MappingUtils.map_mask_to_map_bbox_rcrc(
            map_mask
        )

        # Update pixel to instance mapping
        mask_indices = np.where(map_mask)
        pixel_indices = set()
        for row, col in zip(mask_indices[0], mask_indices[1]):
            pixel_index = MappingUtils.coord_to_index(row, col)
            pixel_indices.add(pixel_index)

            if pixel_index not in self._index_to_instances:
                self._index_to_instances[pixel_index] = set()
            self._index_to_instances[pixel_index].add(instance_id)

        # Update instance to pixel mapping
        self._instance_to_indexes[instance_id] = pixel_indices

    # ==================== modify group ====================

    def add_group(self, group_data: GroupData) -> None:
        """Add group to region"""
        if group_data.instance_ids is None or len(group_data.instance_ids) == 0:
            return

        group_id = group_data.group_id
        for instance_id in group_data.instance_ids:
            instance_data = self.instance_manager.get_instance(instance_id)
            instance_point_cloud = instance_data.point_cloud
            region_mask = MappingUtils.point_cloud_to_map_mask(instance_point_cloud)
            self.add_group_in_map_mask(group_id, region_mask)

    def update_group(self, group_id: int) -> None:
        """Update group information in spatial index based on group point cloud"""
        group_data: GroupData = self.group_manager.groups[group_id]
        if group_data.instance_ids is None or len(group_data.instance_ids) == 0:
            return

        for instance_id in group_data.instance_ids:
            instance_data: InstanceData = self.instance_manager.instances[instance_id]
            if instance_data.point_cloud is None or len(instance_data.point_cloud) == 0:
                continue

            point_cloud = instance_data.point_cloud
            region_mask = MappingUtils.point_cloud_to_map_mask(point_cloud)
            self.add_group_in_map_mask(group_id, region_mask)

    def add_group_in_map_mask(self, group_id: int, region_mask: np.ndarray) -> None:
        """Add group to region"""
        mask_indices = np.where(region_mask)
        for row, col in zip(mask_indices[0], mask_indices[1]):
            pixel_index = MappingUtils.coord_to_index(row, col)

            # Update pixel to group mapping
            if pixel_index not in self._index_to_groups:
                self._index_to_groups[pixel_index] = set()
            self._index_to_groups[pixel_index].add(group_id)

            # Update group to pixel mapping
            if group_id not in self._group_to_indexes:
                self._group_to_indexes[group_id] = set()
            self._group_to_indexes[group_id].add(pixel_index)

    def remove_group(self, group_id: int) -> None:
        """Remove group from region"""
        if group_id not in self._group_to_indexes:
            return

        pixel_indices = self._group_to_indexes[group_id]
        for pixel_index in pixel_indices:
            if pixel_index in self._index_to_groups:
                self._index_to_groups[pixel_index].remove(group_id)

        del self._group_to_indexes[group_id]

    # ==================== completely rebuild index from instance_manager and group_manager ====================

    def rebuild_from_managers(self) -> None:
        """
        Completely rebuild spatial index from instance_manager and group_manager data.
        This method clears all current index data, then reloads all instances and groups from managers,
        rebuilding the complete spatial index structure.
        """
        self.logger.info("Starting to rebuild spatial index from managers...")

        # Clear all existing index data
        self._instance_to_indexes.clear()
        self._index_to_instances.clear()
        self._instance_bboxes.clear()
        self._index_to_groups.clear()
        self._group_to_indexes.clear()

        # Rebuild instance index
        self._rebuild_instance_index()

        # Rebuild group index
        self._rebuild_group_index()

        self.logger.info(
            f"Spatial index rebuild completed. Instances: {len(self._instance_to_indexes)}, Groups: {len(self._group_to_indexes)}"
        )

    def _rebuild_instance_index(self) -> None:
        """Rebuild instance index from instance_manager"""
        instance_ids = self.instance_manager.instance_ids
        self.logger.debug(
            f"Rebuilding instance index, total {len(instance_ids)} instances"
        )

        for instance_id in instance_ids:
            instance_data = self.instance_manager.get_instance(instance_id)
            if instance_data is None:
                self.logger.warning(
                    f"Instance {instance_id} does not exist in manager, skipping"
                )
                continue

            if instance_data.point_cloud is None or len(instance_data.point_cloud) == 0:
                self.logger.warning(
                    f"Instance {instance_id} has no point cloud data, skipping"
                )
                continue

            # Use existing add_instance method to rebuild index
            self.add_instance(instance_data)

    def _rebuild_group_index(self) -> None:
        """Rebuild group index from group_manager"""
        group_ids = self.group_manager.group_ids
        self.logger.debug(f"Rebuilding group index, total {len(group_ids)} groups")

        for group_id in group_ids:
            group_data = self.group_manager.get_group(group_id)
            if group_data is None:
                self.logger.warning(
                    f"Group {group_id} does not exist in manager, skipping"
                )
                continue

            if group_data.instance_ids is None or len(group_data.instance_ids) == 0:
                self.logger.warning(f"Group {group_id} has no instance data, skipping")
                continue

            # Use existing add_group method to rebuild index
            self.add_group(group_data)
