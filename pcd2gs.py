"""Test scripts for point cloud → Gaussian estimation pipeline.

Usage:
    PYTHONPATH=src conda run -n glmap python pcd2gs.py basic
    PYTHONPATH=src conda run -n glmap python pcd2gs.py merge
    PYTHONPATH=src conda run -n glmap python pcd2gs.py          # run both
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gaussian import (
    DiffGaussianCamera,
    GaussianEstimator,
    render_gaussians,
    write_gaussian_ply,
)

ROOT = Path(__file__).parent


def read_ply(path):
    """Read PLY (binary or ASCII) with (x, y, z, red, green, blue)."""
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.float32)
    if colors.max() > 1.0:
        colors /= 255.0
    return positions, colors


def _render(gd, output_path):
    """Render a GaussianData and save to output_path."""
    means = gd.means
    centroid = means.mean(0)
    extent = np.ptp(means, axis=0)
    print(f"    centroid: {centroid}, extent: {extent}")

    distance = float(extent.max()) * 2.0
    eye = centroid.copy()
    eye[2] -= distance

    camera = DiffGaussianCamera.create(
        eye=eye,
        look_at=centroid,
        up_world=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        width=640,
        height=480,
        fov_deg=60.0,
    )

    img = render_gaussians(
        camera=camera,
        means3D=means,
        sh_colors=gd.f_dc,
        opacities=gd.opacities,
        scales=gd.scales,
        rotations=gd.rotations,
        bg_color=(1.0, 1.0, 1.0),
    )
    Image.fromarray(img).save(output_path)
    print(f"    Rendered → {output_path}")


def test_basic():
    """Basic pipeline: read PCD → estimate Gaussians → export & render."""
    output_dir = ROOT / "outputs" / "pcd2gs_basic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Read & center ---
    positions, colors = read_ply(ROOT / "data" / "point_cloud_bed.ply")
    center = positions.mean(0)
    positions -= center
    print(f"[basic] Loaded {len(positions)} points, centered")

    # --- 2. Estimate ---
    estimator = GaussianEstimator()
    gd = estimator.estimate(positions, colors, True)
    print(f"[basic] Estimated {gd.num_gaussians} Gaussians")

    # --- 3. Export & render ---
    write_gaussian_ply(output_dir / "gaussians.ply", gd)
    print(f"[basic] Exported → {output_dir / 'gaussians.ply'}")
    _render(gd, output_dir / "render.png")


def test_merge():
    """Merge pipeline: split PCD into 2 overlapping halves → estimate each → merge."""
    output_dir = ROOT / "outputs" / "pcd2gs_merge"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Read & center ---
    positions, colors = read_ply(ROOT / "data" / "point_cloud_bed.ply")
    center = positions.mean(0)
    positions -= center
    print(f"[merge] Loaded {len(positions)} points, centered")

    # --- 2. Split into two overlapping halves ---
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    x_mid = (x_min + x_max) / 2
    overlap = (x_max - x_min) * 0.25  # 25% overlap
    mask_a = positions[:, 0] <= x_mid + overlap
    mask_b = positions[:, 0] >= x_mid - overlap
    pos_a, col_a = positions[mask_a], colors[mask_a]
    pos_b, col_b = positions[mask_b], colors[mask_b]
    print(
        f"[merge] Split into 2 halves (overlap={overlap:.3f}m)\n"
        f"    A: {len(pos_a)} pts, x in [{pos_a[:, 0].min():.3f}, {pos_a[:, 0].max():.3f}]\n"
        f"    B: {len(pos_b)} pts, x in [{pos_b[:, 0].min():.3f}, {pos_b[:, 0].max():.3f}]"
    )

    # --- 3. Estimate each half ---
    estimator = GaussianEstimator()
    gd_a = estimator.estimate(pos_a, col_a, False)
    gd_b = estimator.estimate(pos_b, col_b, False)
    print(
        f"[merge] Estimated\n"
        f"    A: {gd_a.num_gaussians} Gaussians\n"
        f"    B: {gd_b.num_gaussians} Gaussians"
    )

    # --- 4. Merge ---
    gd_merged = estimator.merge_gaussian_data(gd_a, gd_b)
    naive_sum = gd_a.num_gaussians + gd_b.num_gaussians
    print(
        f"[merge] Merged: {gd_merged.num_gaussians} Gaussians "
        f"(naive concat: {naive_sum}, removed {naive_sum - gd_merged.num_gaussians})"
    )

    # --- 5. Export & render ---
    write_gaussian_ply(output_dir / "gaussians_a.ply", gd_a)
    write_gaussian_ply(output_dir / "gaussians_b.ply", gd_b)
    write_gaussian_ply(output_dir / "gaussians_merged.ply", gd_merged)
    print(f"[merge] Exported PLY files → {output_dir}")
    _render(gd_merged, output_dir / "render_merged.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("basic", "all"):
        test_basic()
    if cmd in ("merge", "all"):
        test_merge()
