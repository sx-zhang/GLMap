import logging
import time
from typing import Dict, List, Optional, Set

import numpy as np

from ..fusion.merge_instance import renew_instance_description
from .data_structures import InstanceData


class InstanceManager:
    """Instance manager"""

    next_instance_id: int = int(time.time() * 1000)

    @classmethod
    def get_next_instance_id(cls) -> int:
        """Create timestamp-based instance ID"""
        instance_id = cls.next_instance_id
        cls.next_instance_id += 1
        return instance_id

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.instances: Dict[int, InstanceData] = {}
        self.category_to_instances: Dict[str, Set[int]] = {}

    @property
    def categories(self) -> List[str]:
        """Return all categories"""
        return list(self.category_to_instances.keys())

    @property
    def instance_ids(self) -> List[int]:
        """Return all instance IDs"""
        return list(self.instances.keys())

    @property
    def instance_datas(self) -> List[InstanceData]:
        """Return all instance data"""
        return list(self.instances.values())

    # ==================== instance simple add/remove ====================

    def get_instance(self, instance_id: int) -> Optional[InstanceData]:
        """Get instance data"""
        return self.instances.get(instance_id)

    def add_instance(self, instance_data: InstanceData) -> int:
        """Add instance and return instance ID"""
        instance_id = instance_data.instance_id
        if instance_id in self.instances:
            raise ValueError(f"Instance ID {instance_id} already exists")
        self.instances[instance_id] = instance_data
        self.category_to_instances.setdefault(instance_data.category, set()).add(
            instance_id
        )
        return instance_id

    def remove_instance(self, instance_id: int) -> bool:
        """Remove instance"""
        if instance_id not in self.instances:
            return False
        instance_data = self.instances.pop(instance_id)
        self.category_to_instances[instance_data.category].remove(instance_id)
        return True

    # ==================== instance complex modifications ====================

    def _add_category_description_for_existing_instance(
        self, instance_id: int, category: str, description: str
    ) -> None:
        """Add description"""
        instance: InstanceData = self.get_instance(instance_id)
        if instance.is_category_description_queue_full:
            all_categories = instance.categories + [category]
            all_descriptions = instance.descriptions + [description]
            history_categories_descriptions = "\n".join(
                [
                    f"category: {category}, description: {description}"
                    for category, description in zip(all_categories, all_descriptions)
                ]
            )
            result = renew_instance_description(
                history_categories_descriptions=history_categories_descriptions
            )
            if result and "category" in result and "description" in result:
                instance.renew_categories_descriptions(
                    new_category=result["category"],
                    new_description=result["description"],
                )
        else:
            instance.add_category_description(category, description)

    def update_existing_instances(
        self,
        existing_instance_id: int,
        input_instance_data: InstanceData,
        gaussian_estimator=None,
    ) -> None:
        """Update instance"""
        existing_instance = self.get_instance(existing_instance_id)
        if existing_instance is None:
            self.logger.error(
                "Failed to update instance %d: instance not found", existing_instance_id
            )
            raise ValueError(f"Instance {existing_instance_id} not found")

        existing_instance.update_point_cloud(
            np.concatenate(
                (existing_instance.point_cloud, input_instance_data.point_cloud), axis=0
            )
        )

        # Merge camera views
        existing_instance.camera_views.extend(input_instance_data.camera_views)

        existing_gd = existing_instance.gaussian_data
        input_gd = input_instance_data.gaussian_data

        if (
            existing_gd is not None
            and input_gd is not None
            and gaussian_estimator is not None
        ):
            merged = gaussian_estimator.merge_gaussian_data(existing_gd, input_gd)
            existing_instance.update_gaussian_data(merged)
        elif input_gd is not None:
            existing_instance.update_gaussian_data(input_gd)

        for category, description in zip(
            input_instance_data.categories, input_instance_data.descriptions
        ):
            self._add_category_description_for_existing_instance(
                existing_instance_id, category, description
            )

    # ==================== instance persistence ====================

    def to_dict(self) -> Dict[int, dict]:
        """Convert to dictionary representation for serialization"""
        return {
            instance_id: instance_data.to_dict()
            for instance_id, instance_data in self.instances.items()
        }

    @staticmethod
    def from_dict(data: Dict[int, dict]) -> "InstanceManager":
        """Load instance data from dictionary"""
        instance_manager = InstanceManager()
        for instance_id, instance_data in data.items():
            instance = InstanceData.from_dict(instance_data)
            instance_manager.add_instance(instance)
        return instance_manager
