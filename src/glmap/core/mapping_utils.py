from typing import Tuple

import numpy as np

from ..config import GLMapConfig


class MappingUtils:
    """Spatial index manager"""

    MAP_HEIGHT = GLMapConfig.MAP_HEIGHT
    MAP_WIDTH = GLMapConfig.MAP_WIDTH
    # self.map_center is the pixel coordinate (row, col) corresponding to (0,0) in episodic coordinate system
    MAP_CENTER = (MAP_HEIGHT // 2, MAP_WIDTH // 2)
    MAP_TOTAL_PIXELS = MAP_HEIGHT * MAP_WIDTH
    MAP_RESOLUTION = GLMapConfig.MAP_RESOLUTION  # meters/pixel

    @classmethod
    def coord_to_index(cls, row: int, col: int) -> int:
        """Convert (row, col) coordinate to 1D index"""
        return row * cls.MAP_WIDTH + col

    @classmethod
    def index_to_coord(cls, index: int) -> Tuple[int, int]:
        """Convert 1D index to (row, col) coordinate"""
        return divmod(index, cls.MAP_WIDTH)

    @classmethod
    def index_to_coord_batch(cls, indices: np.ndarray) -> np.ndarray:
        """Convert 1D indices to (row, col) coordinates"""
        return np.array([cls.index_to_coord(index) for index in indices])

    @classmethod
    def point_cloud_to_map_bbox_rcrc(cls, pc: np.ndarray) -> np.ndarray:
        # Get extremes in pixel space
        rc = cls.episodic_position_to_map_coord_batch(pc[:, :2])  # (N,2)  int array

        # Handle empty point cloud case
        if rc.size == 0:
            return np.array([0, 0, 0, 0], dtype=int)

        min_row, min_col = rc.min(axis=0)
        max_row, max_col = rc.max(axis=0)

        # Make bbox contain complete pixel cells, then expand by half cell
        min_row, min_col = int(np.floor(min_row)), int(np.floor(min_col))
        max_row, max_col = int(np.ceil(max_row)), int(np.ceil(max_col))

        # Prevent out of bounds
        min_row = np.clip(min_row, 0, cls.MAP_HEIGHT - 1)
        max_row = np.clip(max_row, 0, cls.MAP_HEIGHT - 1)
        min_col = np.clip(min_col, 0, cls.MAP_WIDTH - 1)
        max_col = np.clip(max_col, 0, cls.MAP_WIDTH - 1)

        return np.array([min_row, min_col, max_row, max_col], dtype=int)

    @classmethod
    def map_mask_to_map_bbox_rcrc(cls, map_mask: np.ndarray) -> np.ndarray:
        # Get extremes in pixel space
        rc = np.argwhere(map_mask)  # (N,2)  int array

        # Handle empty mask case
        if rc.size == 0:
            # Return an invalid bbox indicating no valid pixels
            return np.array([0, 0, 0, 0], dtype=int)

        min_row, min_col = rc.min(axis=0)
        max_row, max_col = rc.max(axis=0)

        # Make bbox contain complete pixel cells, then expand by half cell
        min_row, min_col = int(np.floor(min_row)), int(np.floor(min_col))
        max_row, max_col = int(np.ceil(max_row)), int(np.ceil(max_col))

        # Prevent out of bounds
        min_row = np.clip(min_row, 0, cls.MAP_HEIGHT - 1)
        max_row = np.clip(max_row, 0, cls.MAP_HEIGHT - 1)
        min_col = np.clip(min_col, 0, cls.MAP_WIDTH - 1)
        max_col = np.clip(max_col, 0, cls.MAP_WIDTH - 1)

        return np.array([min_row, min_col, max_row, max_col], dtype=int)

    @classmethod
    def point_cloud_to_map_mask(cls, point_cloud: np.ndarray) -> np.ndarray:
        """Convert instance_point_cloud [x,y,z] in meters to mask based on map pixels"""
        episodic_coords = point_cloud[:, :2]  # Array of shape (N, 2), containing (x, y)
        map_coords = cls.episodic_position_to_map_coord_batch(
            episodic_coords
        )  # Convert to (row, col)

        # Validate coordinate validity
        valid_mask = (
            (map_coords[:, 1] >= 0)  # col >= 0
            & (map_coords[:, 1] < cls.MAP_WIDTH)  # col < width
            & (map_coords[:, 0] >= 0)  # row >= 0
            & (map_coords[:, 0] < cls.MAP_HEIGHT)  # row < height
        )
        valid_map_coords = map_coords[valid_mask]

        # Set corresponding pixels to True
        region_mask = np.zeros((cls.MAP_HEIGHT, cls.MAP_WIDTH), dtype=bool)
        region_mask[valid_map_coords[:, 0], valid_map_coords[:, 1]] = True  # [row, col]
        return region_mask

    @classmethod
    def point_cloud_to_map_mask_with_height_filting(
        cls,
        episodic_point_cloud: np.ndarray,
        min_height: float,
        max_height: float,
    ) -> np.ndarray:
        """
        Convert point cloud in world coordinate system to obstacle map mask
        """
        height_mask = (episodic_point_cloud[:, 2] >= min_height) & (
            episodic_point_cloud[:, 2] <= max_height
        )
        episodic_point_cloud = episodic_point_cloud[height_mask]
        return cls.point_cloud_to_map_mask(episodic_point_cloud)

    @classmethod
    def episodic_position_to_map_coord(
        cls, episodic_x: float, episodic_y: float
    ) -> Tuple[int, int]:
        """
        Convert world coordinate system coordinates (x, y) in meters to map coordinates (row, col)
        """
        # First convert meters to pixel offset, then add map center offset
        col = (
            int(episodic_x / cls.MAP_RESOLUTION) + cls.MAP_CENTER[1]
        )  # col component of map_center
        row = (
            int(episodic_y / cls.MAP_RESOLUTION) + cls.MAP_CENTER[0]
        )  # row component of map_center
        return (row, col)

    @classmethod
    def episodic_position_to_map_coord_batch(
        cls, episodic_position: np.ndarray
    ) -> np.ndarray:
        """
        Batch convert world coordinate system coordinates (x, y) in meters to map coordinates (row, col)
        """
        if episodic_position.ndim != 2 or episodic_position.shape[1] != 2:
            raise ValueError("episodic_coords must be an array of shape (N, 2)")

        # First convert meters to pixel offset, then add map center offset
        map_coords = (episodic_position / cls.MAP_RESOLUTION).astype(int)
        map_coords += np.array(cls.MAP_CENTER, dtype=int)
        # Swap x and y to match (row, col) format
        return map_coords[:, [1, 0]]

    @classmethod
    def map_coord_to_episodic_position(cls, row: int, col: int) -> Tuple[float, float]:
        """
        Convert map coordinates (row, col) to world coordinate system coordinates (x, y) in meters
        """
        # First subtract map center offset, then convert pixels to meters
        episodic_x = (col - cls.MAP_CENTER[1]) * cls.MAP_RESOLUTION
        episodic_y = (row - cls.MAP_CENTER[0]) * cls.MAP_RESOLUTION
        return (episodic_x, episodic_y)

    @classmethod
    def map_coord_to_episodic_position_batch(cls, map_coords: np.ndarray) -> np.ndarray:
        """
        Batch convert map coordinates (row, col) to world coordinate system coordinates (x, y) in meters
        """
        if map_coords.ndim != 2 or map_coords.shape[1] != 2:
            raise ValueError("map_coords must be an array of shape (N, 2)")

        # First subtract map center offset, then convert pixels to meters
        episodic_coords = (map_coords - np.array(cls.MAP_CENTER, dtype=int)).astype(
            float
        )
        episodic_coords *= cls.MAP_RESOLUTION
        # Swap row and col to match (episodic_x, episodic_y) format
        return episodic_coords[:, [1, 0]]
