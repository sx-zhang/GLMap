import logging

logger = logging.getLogger(__name__)

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import IOConfig
from ..glmap import GLMap


def save_gl_map(
    gl_map: GLMap,
    file_path: Path,
    compress: bool = False,
) -> bool:
    """
    Save map to file
    """
    file_path = Path(file_path)
    # Ensure file extension is .pkl
    if not file_path.suffix.lower() == ".pkl":
        file_path = file_path.with_suffix(".pkl")

    dir_path = file_path.parent
    if dir_path and not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    try:
        # Convert to dictionary representation
        data = gl_map.to_dict()

        # Add metadata
        data["metadata"] = {
            "version": IOConfig.SERIALIZATION_VERSION,
            "compress": compress,
            "timestamp": (
                os.path.getctime(file_path) if os.path.exists(file_path) else None
            ),
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        }

        # Use Pickle to save to file
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Map saved to: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save map: {e}")
        return False


def load_gl_map(file_path: str) -> Optional[GLMap]:
    """
    Load map from file

    Args:
        file_path: Load file path

    Returns:
        Loaded map, returns None if loading fails
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return None

    try:
        # Load data
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Check version compatibility
        metadata = data.get("metadata", {})
        file_version = metadata.get("version", "unknown")

        if file_version != IOConfig.SERIALIZATION_VERSION:
            logger.warning(
                f"Warning: File version {file_version} may be incompatible with current version {IOConfig.SERIALIZATION_VERSION}"
            )

        # Create GLMap instance from dictionary
        gl_map = GLMap.from_dict(data)
        logger.info(f"Map loaded from {file_path} (version: {file_version})")
        return gl_map

    except Exception as e:
        logger.error(f"Failed to load map: {e}")
        return None


def get_gl_map_info(file_path: str) -> Optional[Dict[str, Any]]:
    """Get map file information"""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return None

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        metadata = data.get("metadata", {})

        info = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "version": metadata.get("version", "unknown"),
            "timestamp": metadata.get("timestamp"),
            "shape": data.get("shape", "unknown"),
            "num_instances": len(data.get("instances", {})),
            "num_groups": len(data.get("groups", {})),
        }

        return info

    except Exception as e:
        logger.error(f"Failed to get file information: {e}")
        return None


def export_gl_map_to_text(gl_map: GLMap, file_path: str) -> bool:
    """Export map to readable text format"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("=== GLMap Text Export ===\n\n")
            f.write(f"Map size: {gl_map.shape}\n")
            f.write(f"Number of instances: {len(gl_map.instances)}\n")
            f.write(f"Number of groups: {len(gl_map.group_manager.groups)}\n")
            f.write(f"Number of obstacles: {len(gl_map.obstacle_coords)}\n\n")

            f.write("=== Instance List ===\n")
            for instance_id, instance in gl_map.instances.items():
                f.write(f"Instance {instance_id}:\n")
                f.write(f"  Category: {instance.category}\n")
                f.write(f"  Description: {instance.description}\n")
                f.write(f"  Bounding box: {instance.map_bbox_rcrc}\n\n")

            f.write("=== Group List ===\n")
            for group_id, group in gl_map.group_manager.groups.items():
                f.write(f"Group {group_id}:\n")
                f.write(f"  Description: {group.description}\n")
                f.write(f"  Contains instances: {group.instance_ids}\n\n")

        logger.info(f"Map exported to text format: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export text format: {e}")
        return False
