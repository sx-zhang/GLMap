from typing import Any, Dict, List


def exclude_objects_by_category(
    cleaned: Dict[str, Any], exclude_category: List[str]
) -> Dict[str, Any]:
    """
    Filter objects of specified categories from cleaned data
    """
    # Copy original data to avoid modifying input
    filtered = {**cleaned}

    # Filter objects list, keep objects not in exclusion categories
    filtered_objects = [
        obj for obj in cleaned["objects"] if obj.get("category") not in exclude_category
    ]
    filtered["objects"] = filtered_objects

    # Get remaining object IDs
    remaining_ids = {obj["instance_id"] for obj in filtered_objects}

    # Filter instances in groups, keep only existing object IDs
    filtered_groups = []
    for group in cleaned["groups"]:
        filtered_instances = [
            instance_id
            for instance_id in group["instances"]
            if instance_id in remaining_ids
        ]
        # Only keep groups with at least two objects (meeting original requirements)
        if len(filtered_instances) >= 2:
            filtered_group = {**group}
            filtered_group["instances"] = filtered_instances
            filtered_groups.append(filtered_group)
    filtered["groups"] = filtered_groups

    return filtered
