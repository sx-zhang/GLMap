import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
from scipy.spatial.transform import Rotation

from gaussian import (
    DiffGaussianCamera,
    GaussianData,
    GaussianEstimator,
    render_gaussians,
    write_gaussian_ply,
)
from mllm.ollama_mllm import OllamaMLLM

from .config import CameraConfig, GaussianEstimatorConfig, GLMapConfig, MLLMConfig
from .core.data_structures import CameraView, GroupData, InstanceData
from .core.group_manager import GroupManager
from .core.instance_manager import InstanceManager
from .core.mapping_utils import MappingUtils
from .core.spatial_index import SpatialIndex
from .fusion.merge_instance import check_instance_should_merge


class GLMap:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Basic geometric information
        self._obstacle_map: np.ndarray = np.zeros(
            (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH), dtype=bool
        )
        self._last_agent_coord: Optional[Tuple[int, int]] = None
        self._last_agent_ahead_coord: Optional[Tuple[int, int]] = None
        self._agent_coord_path: List[Tuple[int, int]] = []

        # Core semantic information
        self.instance_manager = InstanceManager()
        self.group_manager = GroupManager(instance_manager=self.instance_manager)
        self.spatial_index = SpatialIndex(
            instance_manager=self.instance_manager,
            group_manager=self.group_manager,
        )

        # External models
        self.gaussian_estimator = GaussianEstimator(
            voxel_size=GaussianEstimatorConfig.VOXEL_SIZE,
            min_points_per_voxel=GaussianEstimatorConfig.MIN_POINTS_PER_VOXEL,
            covariance_regularization=GaussianEstimatorConfig.COVARIANCE_REGULARIZATION,
            init_opacity=GaussianEstimatorConfig.INIT_OPACITY,
            merge_enabled=GaussianEstimatorConfig.MERGE_ENABLED,
            merge_lambda_sigma=GaussianEstimatorConfig.MERGE_LAMBDA_SIGMA,
            merge_lambda_c=GaussianEstimatorConfig.MERGE_LAMBDA_C,
            merge_tau=GaussianEstimatorConfig.MERGE_TAU,
            merge_neighbor_dist=GaussianEstimatorConfig.MERGE_NEIGHBOR_DIST,
        )
        self.mllm = OllamaMLLM(model_id=MLLMConfig.MODEL_ID)

    # ==================== @property ====================
    @property
    def shape(self) -> Tuple[int, int]:
        """Return map shape (height, width)"""
        return (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH)

    @property
    def instances(self) -> Dict[int, InstanceData]:
        """Return all instance data"""
        return self.instance_manager.instances

    @property
    def instance_datas(self) -> List[InstanceData]:
        """Return all instance data"""
        return self.instance_manager.instance_datas

    @property
    def group_datas(self) -> List[GroupData]:
        """Return all group data"""
        return list(self.group_manager.group_datas)

    @property
    def classnames(self) -> List[str]:
        """Return all categories"""
        return self.instance_manager.categories

    @property
    def instance_ids(self) -> List[int]:
        """Return all instance IDs"""
        return list(self.instance_manager.instances.keys())

    @property
    def valid_instances_coords(self) -> List[Tuple[int, int]]:
        """Return pixel coordinates of all valid instances"""
        return self.spatial_index.valid_instances_coords

    # ==================== get ====================

    def get_instances_at_episodic_position(
        self, episodic_x: float, episodic_y: float
    ) -> List[InstanceData]:
        """Get instance list at position from world coordinates in meters"""
        instance_ids = self.spatial_index.get_instance_ids_at_episodic_position(
            episodic_x=episodic_x, episodic_y=episodic_y
        )
        return [self.instances[instance_id] for instance_id in instance_ids]

    def get_instances_at_map_coords(self, row: int, col: int) -> List[InstanceData]:
        """Get instance list at position from map coordinates"""
        instance_ids = self.spatial_index.get_instance_ids_at_map_coords(
            row=row, col=col
        )
        return [self.instances[instance_id] for instance_id in instance_ids]

    # ==================== Geometry ====================

    @property
    def obstacle_map(self) -> np.ndarray:
        """Return obstacle map"""
        return self._obstacle_map

    @property
    def obstacle_coords(self) -> List[Tuple[int, int]]:
        obstacle_rows, obstacle_cols = np.where(self._obstacle_map)
        return list(zip(obstacle_rows, obstacle_cols))

    @property
    def last_agent_ahead_coord(self) -> Optional[Tuple[int, int]]:
        """Return pixel coordinates of last agent ahead position"""
        return self._last_agent_ahead_coord

    @property
    def last_agent_coord(self) -> Optional[Tuple[int, int]]:
        """Return pixel coordinates of last agent position"""
        return self._last_agent_coord

    @property
    def agent_coord_path(self) -> List[Tuple[int, int]]:
        """Return agent path"""
        return self._agent_coord_path

    def update_obstacle_map(
        self,
        episodic_point_cloud: np.ndarray,
        min_height: float = GLMapConfig.OBSTACLE_MIN_HEIGHT,
        max_height: float = GLMapConfig.OBSTACLE_MAX_HEIGHT,
    ):
        """Update obstacle map"""
        new_obstacle_map = MappingUtils.point_cloud_to_map_mask_with_height_filting(
            episodic_point_cloud, min_height, max_height
        )
        self._obstacle_map[new_obstacle_map] = True

    def update_last_agent_coord(self, xFyLzU_position: np.ndarray):
        """Update last agent world coordinate"""
        new_coord = MappingUtils.episodic_position_to_map_coord(
            xFyLzU_position[0], xFyLzU_position[1]
        )
        self._last_agent_coord = new_coord
        self._agent_coord_path.append(new_coord)

    def update_last_agent_ahead_coord(self, xFyLzU_position: np.ndarray):
        """Update last agent ahead coordinate"""
        self._last_agent_ahead_coord = MappingUtils.episodic_position_to_map_coord(
            xFyLzU_position[0], xFyLzU_position[1]
        )

    # ==================== Update Instance Content ====================

    def add_instance_from_property(
        self,
        category: str,
        description: str,
        point_cloud: np.ndarray,
        gaussian_data: Optional["GaussianData"],
        camera_view: Optional[CameraView] = None,
    ) -> int:
        """Add instance from detection results"""
        camera_views = [camera_view] if camera_view else None
        instance_data = InstanceData(
            instance_id=self.instance_manager.get_next_instance_id(),
            category=category,
            description=description,
            point_cloud=point_cloud,
            camera_views=camera_views,
            gaussian_data=gaussian_data,
        )

        return self.add_instance(input_instance_data=instance_data)

    def add_instance_without_spatial_check(
        self, input_instance_data: InstanceData
    ) -> int:
        """Add instance and return instance ID"""
        try:
            self.instance_manager.add_instance(instance_data=input_instance_data)
            self.spatial_index.add_instance(instance_data=input_instance_data)
            return input_instance_data.instance_id
        except Exception as e:
            self.logger.error(
                "Failed to add instance %d: %s", input_instance_data.instance_id, str(e)
            )
            raise e

    def add_instance(self, input_instance_data: InstanceData) -> int:
        """Add instance and return instance ID"""
        episodic_point_cloud = input_instance_data.point_cloud
        existing_instance_ids = (
            self.spatial_index.get_instance_ids_at_episodic_point_cloud(
                episodic_point_cloud
            )
        )
        if len(existing_instance_ids) == 0:
            self.logger.debug(
                "Current instance found no connected instances on map, adding directly"
            )
            return self.add_instance_without_spatial_check(
                input_instance_data=input_instance_data
            )

        try:
            self.logger.debug(
                "Found %d connected instances on map for current instance, need to check if merge is required",
                len(existing_instance_ids),
            )
            should_merge = False
            existing_target_id = None

            for existing_id in existing_instance_ids:
                existing_instance: InstanceData = self.instance_manager.get_instance(
                    existing_id
                )
                if existing_instance is None:
                    continue
                should_merge = check_instance_should_merge(
                    existing_instance,
                    input_instance_data,
                )
                if should_merge:
                    existing_target_id = existing_id
                    break

            if should_merge and existing_target_id is not None:
                self.logger.info(
                    "New instance is similar to map instance %d, merging",
                    existing_target_id,
                )
                self.instance_manager.update_existing_instances(
                    existing_target_id, input_instance_data, self.gaussian_estimator
                )
                self.spatial_index.update_instance(existing_target_id)
                return existing_target_id
            else:
                self.logger.debug(
                    "Found instance is not similar to any map instance, adding as new instance"
                )
                return self.add_instance_without_spatial_check(
                    input_instance_data=input_instance_data
                )
        except Exception as e:
            self.logger.error(
                "Failed to add instance %d: %s", input_instance_data.instance_id, str(e)
            )
            raise e

    # ==================== Update Group Content ====================

    def add_group_from_property(self, description: str, instance_ids: List[int]) -> int:
        """Add group and return group ID"""
        group_data = GroupData(
            group_id=self.group_manager.get_next_group_id(),
            instance_ids=instance_ids,
            description=description,
        )
        return self.add_group(group_data=group_data)

    def add_group(self, group_data: GroupData) -> int:
        """Add group and return group ID"""
        try:
            self.group_manager.add_group(group_data=group_data)
            self.spatial_index.add_group(group_data=group_data)
            return group_data.group_id
        except Exception as e:
            self.logger.error("Failed to add group %d: %s", group_data.group_id, str(e))
            raise e
        finally:
            if self.group_manager.is_need_merge:
                self.merge_groups()

    def merge_groups(self) -> None:
        """Merge groups"""
        try:
            need_merge_sets = self.group_manager.compute_merge_groups()
            for need_merge_set in need_merge_sets:
                if len(need_merge_set) <= 1:
                    continue

                for group_id in need_merge_set:
                    self.spatial_index.remove_group(group_id)

                new_group_id = self.group_manager.merge_existing_groups(need_merge_set)
                self.spatial_index.add_group(self.group_manager.groups[new_group_id])

        except Exception as e:
            self.logger.error("Failed to merge groups: %s", str(e))
            raise e

    # ==================== Gaussian Rendering ====================

    def render_instances(
        self,
        position_xFyLzU: np.ndarray,
        quaternion: np.ndarray,
        instance_ids: List[int],
        width: int = CameraConfig.DEFAULT_WIDTH,
        height: int = CameraConfig.DEFAULT_HEIGHT,
        fov_deg: float = CameraConfig.FOV_DEG,
        bg_rgb: tuple = (0.0, 0.0, 0.0),
        device: str = "cuda",
    ) -> np.ndarray:
        """Render Gaussians of specified instances at given position and orientation, return RGB image.

        Args:
            position: (3,) Camera position, Agent coordinate system (xFyLzU: X forward, Y left, Z up).
            quaternion: (4,) Camera orientation quaternion, Agent coordinate system, (w, x, y, z) order.
                Default orientation is +X (forward).
            instance_ids: List of instance IDs to render.
            width: Output image width.
            height: Output image height.
            fov_deg: Vertical field of view angle (degrees).
            bg_color: Background color (R, G, B), range [0, 1].
            device: Rendering device.

        Returns:
            (H, W, 3) uint8 RGB image.
        """
        # --- Collect Gaussian data of all specified instances (COLMAP coordinate system) ---
        parts_means = []
        parts_sh_colors = []
        parts_opacities = []
        parts_scales = []
        parts_rotations = []

        for iid in instance_ids:
            inst = self.instance_manager.get_instance(iid)
            if inst is None or inst.gaussian_data is None:
                self.logger.warning("Instance %d has no Gaussian data, skipping", iid)
                continue
            gd: GaussianData = inst.gaussian_data
            parts_means.append(gd.means)
            parts_sh_colors.append(gd.f_dc)
            parts_opacities.append(gd.opacities)
            parts_scales.append(gd.scales)
            parts_rotations.append(gd.rotations)

        if not parts_means:
            self.logger.warning("No renderable Gaussian data")
            return np.zeros((height, width, 3), dtype=np.uint8)

        means = np.concatenate(parts_means, axis=0)
        sh_colors = np.concatenate(parts_sh_colors, axis=0)
        opacities = np.concatenate(parts_opacities, axis=0)
        scales = np.concatenate(parts_scales, axis=0)
        rotations = np.concatenate(parts_rotations, axis=0)

        # --- Camera coordinate transform: Agent (xFyLzU) → COLMAP (xRyDzF) ---
        T = np.array(
            [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            dtype=np.float32,
        )
        pos_render = T @ position_xFyLzU.astype(np.float32)

        R_agent2colmap = Rotation.from_matrix(T)
        R_cam_agent = Rotation.from_quat(
            [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        )
        R_cam_colmap = R_agent2colmap * R_cam_agent * R_agent2colmap.inv()
        quat_colmap = R_cam_colmap.as_quat()  # (x, y, z, w)
        quat_colmap = np.array(
            [quat_colmap[3], quat_colmap[0], quat_colmap[1], quat_colmap[2]]
        )  # → (w, x, y, z)

        # Construct look_at: move forward 1 meter along quaternion direction
        R_mat = Rotation.from_quat(
            [quat_colmap[1], quat_colmap[2], quat_colmap[3], quat_colmap[0]]
        ).as_matrix()
        forward_colmap = R_mat[:, 2]
        look_at_render = pos_render + forward_colmap

        # --- Build camera ---
        camera = DiffGaussianCamera.create(
            eye=pos_render,
            look_at=look_at_render,
            up_world=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            width=width,
            height=height,
            fov_deg=fov_deg,
        )

        # --- Render ---
        return render_gaussians(
            camera=camera,
            means3D=means,
            sh_colors=sh_colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            bg_color=bg_rgb,
            device=device,
        )

    # ==================== Auto-Viewpoint Rendering ====================

    @staticmethod
    def _look_at_to_quaternion(eye: np.ndarray, look_at: np.ndarray) -> np.ndarray:
        """Convert eye + look_at to Agent frame quaternion (w, x, y, z).

        Agent frame: xFyLzU, default forward is +X.
        Quaternion represents yaw rotation around Z axis.
        """
        front = look_at - eye
        yaw = math.atan2(float(front[1]), float(front[0]))
        half = yaw / 2.0
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)

    def _select_best_view_for_instance(
        self, instance_id: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Select best (eye, look_at) for rendering a single instance.

        Uses the camera view with the highest weight as eye position,
        and point cloud centroid as look_at target.
        """
        inst = self.instance_manager.get_instance(instance_id)
        if not inst or not inst.camera_views:
            return None

        if inst.point_cloud is None or len(inst.point_cloud) == 0:
            return None

        best_view = max(inst.camera_views, key=lambda v: v.weight)
        eye = best_view.position
        look_at = np.mean(inst.point_cloud, axis=0)

        return eye, look_at

    def _compute_view_for_group(
        self, instance_ids: List[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute (eye, look_at) for rendering a group of instances.

        Uses bounding box center as look_at target, places camera at a distance
        proportional to the bounding box diagonal, in the direction from the
        average observation position toward the center.
        """
        all_points = []
        camera_positions = []

        for iid in instance_ids:
            inst = self.instance_manager.get_instance(iid)
            if inst:
                if inst.point_cloud is not None and len(inst.point_cloud) > 0:
                    all_points.append(inst.point_cloud)
                for cv in inst.camera_views:
                    camera_positions.append(cv.position)

        if not all_points:
            return None

        all_points = np.concatenate(all_points, axis=0)
        center = np.mean(all_points, axis=0)

        # Bounding box diagonal for distance estimation
        bbox_range = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        bbox_diag = float(np.linalg.norm(bbox_range))
        distance = max(bbox_diag * CameraConfig.AUTO_VIEW_DISTANCE_FACTOR, 1.5)

        # Viewing direction: from average camera position toward center
        if camera_positions:
            avg_cam = np.mean(camera_positions, axis=0)
            direction = center - avg_cam
            direction[2] = 0  # horizontal projection
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = np.array([1.0, 0.0, 0.0])

        # Place camera on the same side as observations, further back
        eye = center - direction * distance
        eye[2] = center[2] + distance * CameraConfig.AUTO_VIEW_ELEVATION_FACTOR

        return eye, center

    def render_instance(
        self,
        instance_id: int,
        width: int = CameraConfig.DEFAULT_WIDTH,
        height: int = CameraConfig.DEFAULT_HEIGHT,
        fov_deg: float = CameraConfig.FOV_DEG,
        bg_rgb: tuple = (0.0, 0.0, 0.0),
        device: str = "cuda",
    ) -> np.ndarray:
        """Render single instance from its best viewpoint.

        Selects the camera view with the highest weight and renders
        the instance looking toward the point cloud centroid.
        """
        result = self._select_best_view_for_instance(instance_id)
        if result is None:
            self.logger.warning(
                "Cannot determine viewpoint for instance %d", instance_id
            )
            return np.zeros((height, width, 3), dtype=np.uint8)

        eye, look_at = result
        quaternion = self._look_at_to_quaternion(eye, look_at)

        return self.render_instances(
            position_xFyLzU=eye,
            quaternion=quaternion,
            instance_ids=[instance_id],
            width=width,
            height=height,
            fov_deg=fov_deg,
            bg_rgb=bg_rgb,
            device=device,
        )

    def render_group(
        self,
        instance_ids: List[int],
        width: int = CameraConfig.DEFAULT_WIDTH,
        height: int = CameraConfig.DEFAULT_HEIGHT,
        fov_deg: float = CameraConfig.FOV_DEG,
        bg_rgb: tuple = (0.0, 0.0, 0.0),
        device: str = "cuda",
    ) -> np.ndarray:
        """Render group of instances with auto-computed viewpoint.

        Computes bounding box from all instances, places camera to see the
        entire group from the observation side.
        """
        result = self._compute_view_for_group(instance_ids)
        if result is None:
            self.logger.warning("Cannot determine viewpoint for group")
            return np.zeros((height, width, 3), dtype=np.uint8)

        eye, look_at = result
        quaternion = self._look_at_to_quaternion(eye, look_at)

        return self.render_instances(
            position_xFyLzU=eye,
            quaternion=quaternion,
            instance_ids=instance_ids,
            width=width,
            height=height,
            fov_deg=fov_deg,
            bg_rgb=bg_rgb,
            device=device,
        )

    # ==================== Gaussian Export ====================

    def export_gaussians(self, output_dir: str) -> List[str]:
        """Export Gaussian parameters of all instances as separate PLY files.

        GaussianData internally stores COLMAP (xRyDzF) coordinate system data, export directly.

        Args:
            output_dir: Output directory path.

        Returns:
            List of exported PLY file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        exported = []

        for iid, inst in self.instances.items():
            if inst.gaussian_data is None:
                continue
            gd = inst.gaussian_data
            if gd.num_gaussians == 0:
                continue

            filename = os.path.join(output_dir, f"instance_{iid}_{inst.category}.ply")
            write_gaussian_ply(filename, gd)
            exported.append(filename)
            self.logger.info(
                "Exported instance %d (%s): %d Gaussians → %s",
                iid,
                inst.category,
                gd.num_gaussians,
                filename,
            )

        self.logger.info(
            "Exported Gaussian data of %d instances to %s", len(exported), output_dir
        )
        return exported

    # ==================== Memory Performance Analysis ====================

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # resident memory
            "vms_mb": memory_info.vms / 1024 / 1024,  # virtual memory
            "percent": process.memory_percent(),  # memory usage percentage
        }

    def get_instance_memory_breakdown(self) -> Dict[str, float]:
        """Get memory breakdown of instance data in MB"""
        total_memory = 0
        breakdown = {
            "point_clouds_mb": 0,
            "other_data_mb": 0,
        }

        for instance_id, instance_data in self.instances.items():
            # Point cloud memory
            if instance_data.point_cloud is not None:
                point_cloud_memory = instance_data.point_cloud.nbytes / 1024 / 1024
                breakdown["point_clouds_mb"] += point_cloud_memory
                total_memory += point_cloud_memory

        breakdown["total_instances_mb"] = total_memory
        breakdown["num_instances"] = len(self.instances)

        return breakdown

    def print_memory_stats(self):
        """Print memory statistics"""
        memory_usage = self.get_memory_usage()
        breakdown = self.get_instance_memory_breakdown()

        print("=== GLMap Memory Usage Statistics ===")
        print(
            f"Total process memory: {memory_usage['rss_mb']:.2f} MB ({memory_usage['percent']:.1f}%)"
        )
        print(f"Total instances: {breakdown['num_instances']}")
        print(f"Total instance data memory: {breakdown['total_instances_mb']:.2f} MB")
        print("Memory breakdown:")
        print(f"  - Point cloud data: {breakdown['point_clouds_mb']:.2f} MB")
        print("==============================")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization"""
        return {
            "version": "1.0",
            "shape": self.shape,
            "obstacle_map": self._obstacle_map.tolist(),
            "last_agent_coord": self._last_agent_coord,
            "last_agent_ahead_coord": self._last_agent_ahead_coord,
            "agent_coord_path": self._agent_coord_path,
            "instances": self.instance_manager.to_dict(),
            "groups": self.group_manager.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GLMap":
        """Create GLMap instance from dictionary"""
        gl_map = cls()

        # Restore geometric information
        gl_map._restore_geometry_from_dict(data)

        # Restore coordinate information
        gl_map._restore_coordinates_from_dict(data)

        # Restore instance and group data
        gl_map._restore_instances_and_groups_from_dict(data)

        # Rebuild spatial index
        gl_map._rebuild_spatial_index()

        return gl_map

    def _restore_geometry_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore geometric information (obstacle map) from dictionary data"""
        serialized_shape = data.get(
            "shape", (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH)
        )

        if serialized_shape != (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH):
            self.logger.info(
                f"Detected map size change: adjusting from {serialized_shape} to {(MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH)}"
            )

            if "obstacle_map" in data:
                self._obstacle_map = self._resize_obstacle_map(
                    np.array(data["obstacle_map"], dtype=bool),
                    serialized_shape,
                    (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH),
                )
            else:
                self._obstacle_map = np.zeros(
                    (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH), dtype=bool
                )
        else:
            if "obstacle_map" in data:
                self._obstacle_map = np.array(data["obstacle_map"], dtype=bool)

    def _resize_obstacle_map(
        self,
        old_map: np.ndarray,
        old_shape: Tuple[int, int],
        new_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Resize obstacle map"""
        new_map = np.zeros(new_shape, dtype=bool)
        old_height, old_width = old_shape
        new_height, new_width = new_shape

        # Calculate copy region
        start_row = max(0, (new_height - old_height) // 2)
        start_col = max(0, (new_width - old_width) // 2)
        end_row = min(new_height, start_row + old_height)
        end_col = min(new_width, start_col + old_width)

        # Calculate source region
        src_start_row = max(0, (old_height - new_height) // 2)
        src_start_col = max(0, (old_width - new_width) // 2)
        src_end_row = min(old_height, src_start_row + (end_row - start_row))
        src_end_col = min(old_width, src_start_col + (end_col - start_col))

        # Copy obstacle data
        new_map[start_row:end_row, start_col:end_col] = old_map[
            src_start_row:src_end_row, src_start_col:src_end_col
        ]

        return new_map

    def _restore_coordinates_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore coordinate information from dictionary data"""
        serialized_shape = data.get(
            "shape", (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH)
        )

        # Restore last agent coordinate
        self._last_agent_coord = self._adjust_coordinate_from_dict(
            data.get("last_agent_coord"), serialized_shape
        )

        # Restore last agent ahead coordinate
        self._last_agent_ahead_coord = self._adjust_coordinate_from_dict(
            data.get("last_agent_ahead_coord"), serialized_shape
        )

        # Restore agent coordinate path
        self._agent_coord_path = self._adjust_coordinate_path_from_dict(
            data.get("agent_coord_path", []), serialized_shape
        )

    def _adjust_coordinate_from_dict(
        self, coord_data: Optional[List[int]], serialized_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Adjust single coordinate to fit new map size"""
        if not coord_data:
            return None

        old_coord = tuple(coord_data)

        if serialized_shape != (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH):
            old_height, old_width = serialized_shape
            new_height, new_width = MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH

            # Calculate coordinate offset
            row_offset = (new_height - old_height) // 2
            col_offset = (new_width - old_width) // 2

            new_row = max(0, min(new_height - 1, old_coord[0] + row_offset))
            new_col = max(0, min(new_width - 1, old_coord[1] + col_offset))
            return (new_row, new_col)
        else:
            return old_coord

    def _adjust_coordinate_path_from_dict(
        self, path_data: List[List[int]], serialized_shape: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Adjust coordinate path to fit new map size"""
        old_path = [tuple(coord) for coord in path_data]

        if serialized_shape != (MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH):
            old_height, old_width = serialized_shape
            new_height, new_width = MappingUtils.MAP_HEIGHT, MappingUtils.MAP_WIDTH

            # Calculate coordinate offset
            row_offset = (new_height - old_height) // 2
            col_offset = (new_width - old_width) // 2

            new_path = []
            for coord in old_path:
                new_row = max(0, min(new_height - 1, coord[0] + row_offset))
                new_col = max(0, min(new_width - 1, coord[1] + col_offset))
                new_path.append((new_row, new_col))
            return new_path
        else:
            return old_path

    def _restore_instances_and_groups_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore instance and group data from dictionary data"""
        # Restore instance data
        if "instances" in data:
            self.instance_manager = InstanceManager.from_dict(data["instances"])
        else:
            self.logger.error("Failed to import instance data")

        # Restore group data
        if "groups" in data:
            self.group_manager = GroupManager.from_dict(
                self.instance_manager, data["groups"]
            )
        else:
            self.logger.error("Failed to import group data")

    def _rebuild_spatial_index(self) -> None:
        """Rebuild spatial index"""
        self.spatial_index = SpatialIndex(
            instance_manager=self.instance_manager,
            group_manager=self.group_manager,
        )
        self.spatial_index.rebuild_from_managers()
