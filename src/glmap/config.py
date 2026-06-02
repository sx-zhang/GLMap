from dataclasses import dataclass
from typing import Optional


@dataclass
class InstanceDataConfig:
    """Instance configuration class"""

    MAX_RGB_PATCHES_PER_INSTANCE: int = 3
    MAX_DESCRIPTIONS_PER_INSTANCE: int = 3


@dataclass
class GLMapConfig:
    """GLMap configuration class"""

    MAP_HEIGHT: int = 1000
    MAP_WIDTH: int = 1000
    MAP_RESOLUTION: float = 0.05  # meters/pixel
    OBSTACLE_MIN_HEIGHT: float = 0.5  # meters
    OBSTACLE_MAX_HEIGHT: float = 0.88  # meters


@dataclass
class GaussianEstimatorConfig:
    """Gaussian estimator configuration class"""

    VOXEL_SIZE: float = 0.02  # voxelization spatial resolution (meters)
    MIN_POINTS_PER_VOXEL: int = (
        5  # minimum points in neighborhood, below this no Gaussian is generated
    )
    COVARIANCE_REGULARIZATION: float = (
        1e-6  # covariance regularization for numerical stability
    )
    INIT_OPACITY: float = 1.0  # initial opacity [0, 1]

    MERGE_ENABLED: bool = False  # whether to merge similar Gaussians after estimation

    MERGE_LAMBDA_SIGMA: float = -20.0  # lambda parameter for merging similar Gaussians
    MERGE_LAMBDA_C: float = 0.4  # weight for color distance in merge metric
    MERGE_TAU: float = -20.0  # curvature factor for adaptive merge threshold
    MERGE_NEIGHBOR_DIST: Optional[float] = (
        None  # search radius for merge neighbors (None = voxel_size)
    )


@dataclass
class InstanceFusionConfig:
    """Instance fusion configuration class"""

    CLOUD_POINT_CONNECTITY_THRESH: float = 0.15  # meters
    CLOUD_POINT_CONNECTITY_COUNT_THRESH: int = (
        2  # minimum number of point pairs within threshold to consider connected
    )

    TEXT_EMBEDDING_MODEL_ID: str = "nomic-embed-text"
    TEXT_EMBEDDING_SIMILARITY_THRESH: float = 0.7

    MERGE_RESULT_LLM_MODEL_ID: str = "qwen3:8b"


@dataclass
class GroupFusionConfig:
    """Group fusion configuration class"""

    GROUP_MERGE_INTERVAL: int = 3

    TEXT_EMBEDDING_MODEL_ID: str = "nomic-embed-text"
    TEXT_EMBEDDING_SIMILARITY_THRESH: float = 0.7

    # LLM model for generating merged descriptions
    MERGE_DESCRIPTION_MODEL_ID: str = "qwen3:8b"


@dataclass
class MLLMConfig:
    """MLLM configuration class"""

    MODEL_ID: str = "gemma3:27b"


@dataclass
class CameraConfig:
    """Camera default parameters"""

    FOV_DEG: float = 60.0
    DEFAULT_WIDTH: int = 640
    DEFAULT_HEIGHT: int = 480
    AUTO_VIEW_DISTANCE_FACTOR: float = 1.2  # bbox diagonal multiplier for group view
    AUTO_VIEW_ELEVATION_FACTOR: float = 0.2  # height offset = distance * factor


@dataclass
class IOConfig:
    """IO configuration class"""

    SERIALIZATION_FORMAT: str = "pickle"
    SERIALIZATION_VERSION: str = "1.0"
