import logging
import time
from typing import Dict, List, Optional, Set

from ..config import GroupFusionConfig
from ..fusion.merge_group import compute_merge_groups, generate_group_description
from .data_structures import GroupData
from .instance_manager import InstanceManager


class GroupManager:
    """Group manager"""

    next_group_id: int = int(time.time() * 1000)

    def __init__(self, instance_manager: InstanceManager):
        self.logger = logging.getLogger(__name__)
        self.groups: Dict[int, GroupData] = {}
        self.instance_manager = instance_manager  # reference to instance manager
        self.instance_to_groups: Dict[int, Set[int]] = {}

    @property
    def group_ids(self) -> List[int]:
        """Return all group IDs"""
        return list(self.groups.keys())

    @property
    def group_datas(self) -> List[GroupData]:
        """Return all group data"""
        return list(self.groups.values())

    @property
    def is_need_merge(self) -> bool:
        """Check if groups need to be merged"""
        return (
            len(self.group_ids) > 0
            and len(self.group_ids) % GroupFusionConfig.GROUP_MERGE_INTERVAL == 0
        )

    @classmethod
    def get_next_group_id(cls) -> int:
        """Create timestamp-based group ID"""
        group_id = cls.next_group_id
        cls.next_group_id += 1
        return group_id

    def add_group(self, group_data: GroupData) -> None:
        """Add group"""
        self.groups[group_data.group_id] = group_data
        for instance_id in group_data.instance_ids:
            self.instance_to_groups.setdefault(instance_id, set()).add(
                group_data.group_id
            )

    def remove_group(self, group_id: int) -> bool:
        """Remove group"""
        if group_id not in self.groups:
            return False
        group = self.groups.pop(group_id)
        for instance_id in group.instance_ids:
            if instance_id not in self.instance_to_groups:
                self.logger.warning(
                    "Instance %d pointed to by group %d does not exist in instance_to_groups",
                    group_id,
                    instance_id,
                )
                continue
            self.instance_to_groups[instance_id].remove(group_id)
        return True

    def get_group(self, group_id: int) -> Optional[GroupData]:
        """Get group data"""
        return self.groups.get(group_id)

    def get_groups_by_instance(self, instance_id: int) -> List[GroupData]:
        """
        Get all groups containing this instance by instance ID
        """
        group_ids = self.instance_to_groups.get(instance_id, set())
        return [self.groups[gid] for gid in group_ids if gid in self.groups]

    def add_to_group(self, group_id: int, instance_id: int) -> bool:
        """Add instance to group"""
        if group_id not in self.groups:
            return False
        group = self.groups[group_id]
        if instance_id in group.instance_ids:
            return False
        group.instance_ids.add(instance_id)
        self.instance_to_groups.setdefault(instance_id, set()).add(group_id)
        return True

    def remove_from_group(self, group_id: int, instance_id: int) -> bool:
        """Remove instance from group"""
        if group_id not in self.groups:
            return False
        group = self.groups[group_id]
        if instance_id not in group.instance_ids:
            return False
        group.instance_ids.remove(instance_id)
        self.instance_to_groups[instance_id].remove(group_id)
        return True

    def compute_merge_groups(self) -> Set[Set[int]]:
        """Compute groups that need to be merged"""
        need_merge_sets = compute_merge_groups(self.group_datas)
        self.logger.info(f"Groups to merge: {need_merge_sets}")
        return need_merge_sets

    def merge_existing_groups(self, need_merge_set: Set[int]) -> int:
        """Merge groups that need to be merged"""
        merged_instance_ids = set()
        for group_id in need_merge_set:
            group = self.groups[group_id]
            merged_instance_ids.update(group.instance_ids)
            self.remove_group(group_id)

        instance_datas = [
            self.instance_manager.get_instance(instance_id)
            for instance_id in merged_instance_ids
        ]

        description = generate_group_description(instance_datas)

        # Update group description based on instance data
        new_group = GroupData(
            group_id=self.get_next_group_id(),
            instance_ids=merged_instance_ids,
            description=description,
        )
        self.add_group(new_group)
        for instance_id in merged_instance_ids:
            self.instance_to_groups.setdefault(instance_id, set()).add(
                new_group.group_id
            )

        return new_group.group_id

    def to_dict(self) -> Dict[int, dict]:
        """Convert to dictionary representation for serialization"""
        return {gid: group.to_dict() for gid, group in self.groups.items()}

    @classmethod
    def from_dict(
        cls, instance_manager: InstanceManager, data: Dict[int, dict]
    ) -> "GroupManager":
        """Load group data from dictionary"""
        group_manager = cls(instance_manager)
        group_manager.groups.clear()
        group_manager.instance_to_groups.clear()

        for gid, group_data in data.items():
            group = GroupData.from_dict(group_data)
            group_manager.add_group(group)

        return group_manager
