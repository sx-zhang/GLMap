import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.coordinate_transforms import xFyLzU_pcd_to_xRyDzF_pcd

C0 = 0.28209479177387814


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB colors to Spherical Harmonics (degree 0) coefficients."""
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: np.ndarray) -> np.ndarray:
    """Convert SH (degree 0) coefficients back to RGB."""
    return sh * C0 + 0.5


def opacity_to_logit(opacity: float) -> float:
    """Convert opacity (0, 1) to logit space."""
    opacity = np.clip(opacity, 1e-6, 1.0 - 1e-6)
    return np.log(opacity / (1.0 - opacity))


# xFyLzU → xRyDzF rotation matrix
_T_AGENT_TO_COLMAP = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
_R_AGENT_TO_COLMAP = R.from_matrix(_T_AGENT_TO_COLMAP)


def agent_to_colmap(means: np.ndarray, rotations: np.ndarray) -> tuple:
    """Convert Gaussian positions and rotations from Agent (xFyLzU) to COLMAP (xRyDzF).

    Args:
        means: (N, 3) positions in Agent frame.
        rotations: (N, 4) quaternions (w,x,y,z) in Agent frame.

    Returns:
        (means_colmap, rotations_colmap) in COLMAP frame.
    """
    means_colmap = xFyLzU_pcd_to_xRyDzF_pcd(means)

    # R_colmap = T @ R_agent
    quats_xyzw = np.stack(
        [rotations[:, 1], rotations[:, 2], rotations[:, 3], rotations[:, 0]],
        axis=1,
    )
    r_colmap = _R_AGENT_TO_COLMAP * R.from_quat(quats_xyzw)
    quats_out = r_colmap.as_quat()  # (x,y,z,w)
    rotations_colmap = np.stack(
        [quats_out[:, 3], quats_out[:, 0], quats_out[:, 1], quats_out[:, 2]],
        axis=1,
    ).astype(np.float32)

    return means_colmap, rotations_colmap
