import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from gaussian.data import GaussianData

from ..config import InstanceDataConfig
from .mapping_utils import MappingUtils


class CameraView:
    """Camera viewpoint with position and observation quality weight."""

    def __init__(self, position: np.ndarray, weight: float):
        self.position = position  # (3,) Agent frame (xFyLzU: X forward, Y left, Z up)
        self.weight = weight  # detection_score × point_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraView":
        return cls(
            position=np.array(data["position"]),
            weight=data["weight"],
        )


class InstanceData:
    """Instance data class"""

    def __init__(
        self,
        instance_id: int,
        category: str,
        description: str,
        point_cloud: np.ndarray,
        camera_views: Optional[List[CameraView]] = None,
        gaussian_data: Optional[GaussianData] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.instance_id = instance_id

        self.categories = [category]
        self.descriptions = [description]

        self.point_cloud = None
        self.point_cloud_updated_timestamp = None
        self.update_point_cloud(new_point_cloud=point_cloud)

        self.camera_views: List[CameraView] = camera_views or []
        self.gaussian_data = gaussian_data

    def update_gaussian_data(self, gaussian_data: GaussianData):
        """Update instance Gaussian parameters"""
        self.gaussian_data = gaussian_data

    @property
    def category(self) -> str:
        """Return main category of instance"""
        return self.categories[0]

    @property
    def description(self) -> str:
        """Return longest instance description"""
        return max(self.descriptions, key=len)

    @property
    def is_category_description_queue_full(self) -> bool:
        """Check if descriptions need to be merged"""
        assert len(self.categories) == len(self.descriptions)
        return (
            len(self.descriptions) >= InstanceDataConfig.MAX_DESCRIPTIONS_PER_INSTANCE
        )

    def update_point_cloud(self, new_point_cloud):
        """Update map mask based on current point cloud"""
        self.point_cloud = new_point_cloud
        self.map_bbox_rcrc = MappingUtils.point_cloud_to_map_bbox_rcrc(self.point_cloud)
        self.point_cloud_updated_timestamp = time.time()

    def add_category_description(self, category: str, description: str):
        """Add description, cannot auto-replace when full"""
        if self.is_category_description_queue_full:
            raise ValueError(
                "Instance description queue is full, cannot add new category and description"
            )
        self.categories.append(category)
        self.descriptions.append(description)

    def renew_categories_descriptions(self, new_category: str, new_description: str):
        """Reset all descriptions"""
        self.categories = [new_category]
        self.descriptions = [new_description]

    # ==================== Persistence ====================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization"""
        return {
            "instance_id": self.instance_id,
            "categories": self.categories,
            "descriptions": self.descriptions,
            "camera_views": [cv.to_dict() for cv in self.camera_views],
            "point_cloud": (
                self.point_cloud.tolist() if self.point_cloud is not None else None
            ),
            "map_bbox_rcrc": self.map_bbox_rcrc,
            "point_cloud_updated_timestamp": self.point_cloud_updated_timestamp,
            "gaussian_data": (
                self.gaussian_data._data.tolist()
                if self.gaussian_data is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstanceData":
        """Create InstanceData instance from dictionary"""
        # Backward compat: old format has camera_position (single array)
        camera_views = None
        if "camera_views" in data and data["camera_views"]:
            camera_views = [CameraView.from_dict(cv) for cv in data["camera_views"]]
        elif "camera_position" in data and data["camera_position"] is not None:
            camera_views = [
                CameraView(position=np.array(data["camera_position"]), weight=0.0)
            ]

        # Create base instance
        instance = cls(
            instance_id=data["instance_id"],
            category=data["categories"][0],
            description=data["descriptions"][0],
            point_cloud=np.array(data["point_cloud"]) if data["point_cloud"] else None,
            camera_views=camera_views,
        )

        # Restore complete data
        if len(data["categories"]) > 1:
            instance.categories = data["categories"]
            instance.descriptions = data["descriptions"]

        # Restore map bounding box
        instance.map_bbox_rcrc = data.get("map_bbox_rcrc", instance.map_bbox_rcrc)
        instance.point_cloud_updated_timestamp = data.get(
            "point_cloud_updated_timestamp", instance.point_cloud_updated_timestamp
        )

        # Restore Gaussian parameters
        if data.get("gaussian_data") is not None:
            instance.gaussian_data = GaussianData(
                np.array(data["gaussian_data"], dtype=np.float32)
            )

        return instance


class GroupData:
    """Group data class"""

    def __init__(self, group_id: int, instance_ids: List[int], description: str):
        self.group_id = group_id
        self.instance_ids = instance_ids
        self.description = description
        self.rgb = None

    def add_instance(self, instance_id: int) -> None:
        """Add instance to group"""
        self.instance_ids.append(instance_id)

    def remove_instance(self, instance_id: int) -> None:
        """Remove instance from group"""
        if instance_id in self.instance_ids:
            self.instance_ids.remove(instance_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization"""
        return {
            "group_id": self.group_id,
            "instance_ids": self.instance_ids,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupData":
        """Create GroupData instance from dictionary"""
        return cls(
            group_id=data["group_id"],
            instance_ids=data["instance_ids"],
            description=data["description"],
        )
