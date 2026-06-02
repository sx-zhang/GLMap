import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from .camera import DiffGaussianCamera


def render_gaussians(
    camera: DiffGaussianCamera,
    means3D: np.ndarray,
    sh_colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    bg_color: tuple = (0.0, 0.0, 0.0),
    scale_modifier: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """Render an RGB image from 3D Gaussians using diff-gaussian-rasterization.

    Args:
        camera: DiffGaussianCamera with extrinsic/intrinsic parameters.
        means3D: (M, 3) Gaussian centers in COLMAP/OpenCV (xRyDzF) coordinates.
        sh_colors: (M, 3) SH degree-0 coefficients.
        opacities: (M,) Opacities in logit space.
        scales: (M, 3) Log-space scales.
        rotations: (M, 4) Quaternions in (w, x, y, z) order.
        bg_color: Background RGB tuple.
        scale_modifier: Scale modifier for rasterization.
        device: Torch device.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    M = means3D.shape[0]
    settings = camera.to_diff_gaussian_settings(
        bg_color=bg_color,
        scale_modifier=scale_modifier,
        sh_degree=0,
        device=device,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    means3D_t = torch.from_numpy(means3D).float().to(device)
    means2D = torch.zeros((M, 3), dtype=torch.float32, device=device)
    # Opacity is in logit space, requires sigmoid activation
    opacities_t = torch.sigmoid(
        torch.from_numpy(opacities).float().to(device)
    ).unsqueeze(-1)
    # Scale is in log space, requires exp activation to restore to physical volume
    scales_t = torch.exp(torch.from_numpy(scales).float().to(device))
    # L2 normalize quaternions to eliminate precision truncation errors from file I/O
    rotations_t = torch.nn.functional.normalize(
        torch.from_numpy(rotations).float().to(device), p=2, dim=-1
    )

    shs = torch.from_numpy(sh_colors).float().to(device)
    shs = shs.unsqueeze(1)  # (M, 1, 3)

    with torch.no_grad():
        color, _ = rasterizer(
            means3D=means3D_t,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacities_t,
            scales=scales_t,
            rotations=rotations_t,
            cov3D_precomp=None,
        )

    color = color.clamp(0.0, 1.0)
    image = (color.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return image
