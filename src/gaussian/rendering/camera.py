from dataclasses import dataclass

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings

FOV_DEG = 60
ASPECT_RATIO = 1.0
DEFAULT_WIDTH, DEFAULT_HEIGHT = 640, 480


@dataclass
class BasicCamera:
    eye: np.ndarray  # (3,)
    front: np.ndarray  # (3,)
    up: np.ndarray  # (3,)
    right: np.ndarray  # (3,)
    width: int
    height: int
    fov_deg: float
    aspect: float

    @classmethod
    def create(
        cls,
        eye: np.ndarray,
        look_at: np.ndarray,
        up_world: np.ndarray,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fov_deg: float = FOV_DEG,
        aspect: float = ASPECT_RATIO,
    ):
        front = (look_at - eye).astype(float)
        front /= np.linalg.norm(front)

        up = np.asarray(up_world, float)
        up -= np.dot(up, front) * front
        up /= np.linalg.norm(up)

        right = np.cross(front, up)
        right /= np.linalg.norm(right)
        return cls(
            eye=eye,
            front=front,
            up=up,
            right=right,
            width=width,
            height=height,
            fov_deg=fov_deg,
            aspect=aspect,
        )

    # Extrinsic 4x4 matrix (world -> camera)
    @property
    def extrinsic(self) -> np.ndarray:
        R_c2w = np.column_stack([self.right, -self.up, self.front])
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ self.eye
        ext = np.eye(4)
        ext[:3, :3] = R_w2c
        ext[:3, 3] = t_w2c
        return ext


@dataclass
class DiffGaussianCamera(BasicCamera):
    """
    Inherits all methods from BasicCamera, only overrides/adds necessary interfaces.
    """

    @property
    def fx(self) -> float:
        """Pixel focal length fx"""
        return self.width / 2.0 / np.tan(np.radians(self.fov_deg) / 2.0)

    @property
    def fy(self) -> float:
        """Pixel focal length fy"""
        return self.height / 2.0 / np.tan(np.radians(self.fov_deg) / 2.0)

    @property
    def tanfovx(self) -> float:
        """tan(0.5 * fovX) required by diff-gaussian"""
        return float(np.tan(np.radians(self.fov_deg) / 2.0))

    @property
    def tanfovy(self) -> float:
        """tan(0.5 * fovY) required by diff-gaussian"""
        return float(np.tan(np.radians(self.fov_deg) / 2.0))

    def to_diff_gaussian_settings(
        self,
        bg_color: tuple = (0.0, 0.0, 0.0),
        scale_modifier: float = 1.0,
        sh_degree: int = 0,
        prefiltered: bool = False,
        debug: bool = False,
        device="cuda",
    ) -> GaussianRasterizationSettings:
        # 4x4 world->camera
        extrinsic = torch.from_numpy(self.extrinsic).float().to(device)

        # OpenGL style projection matrix
        near, far = 0.01, 100.0
        P = torch.zeros((4, 4), device=device)
        P[0, 0] = 1.0 / self.tanfovx
        P[1, 1] = 1.0 / self.tanfovy
        P[2, 2] = far / (far - near)
        P[3, 2] = 1.0
        P[2, 3] = -(far * near) / (far - near)

        # Extrinsic matrix
        Rt = torch.zeros((4, 4), device=device)
        Rt[:3, :3] = extrinsic[:3, :3]
        Rt[:3, 3] = extrinsic[:3, 3]
        Rt[3, 3] = 1.0

        # Complete projection matrix
        projmatrix = P @ Rt

        # Camera position (world coordinates)
        campos = torch.from_numpy(self.eye).float().to(device)

        return GaussianRasterizationSettings(
            image_height=self.height,
            image_width=self.width,
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            bg=torch.tensor(bg_color, dtype=torch.float32, device=device),
            scale_modifier=scale_modifier,
            # [Key modification]: Transpose to adapt to CUDA column-major order, and force contiguous memory layout
            viewmatrix=extrinsic.transpose(0, 1).contiguous(),
            projmatrix=projmatrix.transpose(0, 1).contiguous(),
            sh_degree=sh_degree,
            campos=campos,
            prefiltered=prefiltered,
            debug=debug,
        )

    @classmethod
    def create_diff_gaussian_camera(
        cls,
        eye: np.ndarray,
        look_at: np.ndarray,
        up_world: np.ndarray,
        width: int = 640,
        height: int = 480,
        fov_deg: float = 60.0,
        aspect: float = 1.0,
        **kw,
    ) -> "DiffGaussianCamera":
        return cls.create(
            eye=eye,
            look_at=look_at,
            up_world=up_world,
            width=width,
            height=height,
            fov_deg=fov_deg,
            aspect=aspect,
        )
