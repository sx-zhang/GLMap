from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.coordinate_transforms import (
    xFyLzU_pcd_to_xRyDzF_pcd,
    xRyDzF_pcd_to_xFyLzU_pcd,
)

from .utils import sh_to_rgb

# Field specs: (name, shape_suffix, dtype)
# Aligned with 3DGS PLY format: x, y, z, f_dc_0, f_dc_1, f_dc_2,
#   opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
#
# Internally stored in COLMAP/OpenCV coordinates (xRyDzF).
_GAUSSIAN_FIELDS = [
    ("means", (3,), np.float32),  # Gaussian centers (x, y, z) in COLMAP
    ("f_dc", (3,), np.float32),  # SH DC coefficients (f_dc_0, f_dc_1, f_dc_2)
    ("opacities", (), np.float32),  # logit space
    ("scales", (3,), np.float32),  # log-space scales
    ("rotations", (4,), np.float32),  # quaternions (w,x,y,z) in COLMAP
]

# xFyLzU → xRyDzF rotation matrix (for transforming quaternions)
_T_AGENT_TO_COLMAP = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
_T_COLMAP_TO_AGENT = _T_AGENT_TO_COLMAP.T  # orthogonal, so inverse = transpose

_R_AGENT_TO_COLMAP = R.from_matrix(_T_AGENT_TO_COLMAP)
_R_COLMAP_TO_AGENT = R.from_matrix(_T_COLMAP_TO_AGENT)


class GaussianData:
    """3D Gaussian Splatting attributes backed by a single numpy array.

    **Coordinate system**: Internally stores data in COLMAP/OpenCV (xRyDzF)
    coordinates. Use ``agent_means`` / ``agent_rotations`` to access in the
    Agent (xFyLzU) frame used by GLMap's spatial operations.

    Layout (14 float32 per Gaussian):

    | Offset | Field     | Shape | Meaning                    |
    |--------|-----------|-------|----------------------------|
    | 0      | means     | (3,)  | Center in COLMAP (xRyDzF)  |
    | 3      | f_dc      | (3,)  | SH DC coefficients         |
    | 6      | opacities | ()    | Opacity (logit space)      |
    | 7      | scales    | (3,)  | Log-space scale per axis   |
    | 10     | rotations | (4,)  | Quaternion (w,x,y,z) COLMAP|

    Use ``to_dict()`` / ``from_dict()`` for zero-copy views.
    """

    _FIELDS = _GAUSSIAN_FIELDS
    _STRIDE = sum(int(np.prod(s)) for _, s, _ in _GAUSSIAN_FIELDS)

    def __init__(self, data: np.ndarray):
        self._data = np.ascontiguousarray(data, dtype=np.float32)

    # -- dict conversion (zero-copy views) --

    def to_dict(self) -> dict:
        """Return a dict of numpy array *views* keyed by field name."""
        d = {}
        offset = 0
        for name, shape_suffix, _ in self._FIELDS:
            size = int(np.prod(shape_suffix)) if shape_suffix else 1
            d[name] = self._data[:, offset : offset + size].reshape(-1, *shape_suffix)
            offset += size
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianData":
        """Build from a dict of numpy arrays (keys must match field names)."""
        M = len(d["means"])
        buf = np.empty((M, cls._STRIDE), np.float32)
        offset = 0
        for name, shape_suffix, _ in cls._FIELDS:
            flat = d[name].reshape(M, -1)
            size = flat.shape[1]
            buf[:, offset : offset + size] = flat
            offset += size
        return cls(buf)

    # -- named access (view into underlying buffer) --

    def _view(self, name: str) -> np.ndarray:
        offset = 0
        for n, shape_suffix, _ in self._FIELDS:
            size = int(np.prod(shape_suffix)) if shape_suffix else 1
            if n == name:
                col = self._data[:, offset : offset + size]
                if shape_suffix:
                    return col.reshape(-1, *shape_suffix)
                return col
            offset += size
        raise AttributeError(name)

    # -- COLMAP properties (stored directly) --

    @property
    def means(self) -> np.ndarray:
        """(N, 3) Gaussian centers in COLMAP (xRyDzF)."""
        return self._view("means")

    @property
    def f_dc(self) -> np.ndarray:
        """(N, 3) SH DC coefficients (f_dc_0, f_dc_1, f_dc_2)."""
        return self._view("f_dc")

    @property
    def rgb(self) -> np.ndarray:
        """(N, 3) RGB colors [0-1], computed from SH DC coefficients."""
        return sh_to_rgb(self.f_dc)

    @property
    def opacities(self) -> np.ndarray:
        """(N,) Opacities in logit space."""
        return self._view("opacities").squeeze(axis=1)

    @property
    def scales(self) -> np.ndarray:
        """(N, 3) Log-space scales."""
        return self._view("scales")

    @property
    def rotations(self) -> np.ndarray:
        """(N, 4) Quaternions (w,x,y,z) in COLMAP frame."""
        return self._view("rotations")

    @property
    def num_gaussians(self) -> int:
        return self._data.shape[0]

    # -- Agent properties (computed on-the-fly) --

    @property
    def agent_means(self) -> np.ndarray:
        """(N, 3) Gaussian centers in Agent (xFyLzU) frame."""
        return xRyDzF_pcd_to_xFyLzU_pcd(self.means)

    @property
    def agent_rotations(self) -> np.ndarray:
        """(N, 4) Quaternions (w,x,y,z) in Agent frame."""
        return _transform_quats(self.rotations, _R_COLMAP_TO_AGENT)


def _transform_quats(quats_wxyz: np.ndarray, r_transform: R) -> np.ndarray:
    """Transform quaternions by a rotation. Input/output in (w,x,y,z)."""
    quats_xyzw = np.stack(
        [quats_wxyz[:, 1], quats_wxyz[:, 2], quats_wxyz[:, 3], quats_wxyz[:, 0]],
        axis=1,
    )
    r_out = r_transform * R.from_quat(quats_xyzw)
    quats_out = r_out.as_quat()  # (x,y,z,w)
    return np.stack(
        [quats_out[:, 3], quats_out[:, 0], quats_out[:, 1], quats_out[:, 2]],
        axis=1,
    ).astype(np.float32)
