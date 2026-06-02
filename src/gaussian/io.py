import os

import numpy as np

from .data import GaussianData

binary_little_endian_ply_header = """ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""


def write_gaussian_ply(output_filename: str, gaussian_data: GaussianData):
    """Write GaussianData to a PLY file in binary_little_endian format.

    The underlying buffer layout of GaussianData (14 float32 per point) matches
    the 3DGS PLY format exactly, so this writes directly without transformation.

    Attribute order (14 float32 values):
    (x, y, z)                                    # COLMAP coordinate system
    (f_dc_0, f_dc_1, f_dc_2)                     # Zero-order SH color coefficients
    (opacity)                                     # logit space, requires sigmoid to restore to [0,1]
    (scale_0, scale_1, scale_2)                   # log space, requires exp to restore to actual scale
    (rot_0, rot_1, rot_2, rot_3)                  # Normalized quaternion (w, x, y, z)
    """
    num_points = gaussian_data.num_gaussians
    header = binary_little_endian_ply_header.format(num_points=num_points)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(gaussian_data._data.astype(np.float32).tobytes())


def read_gaussian_ply(filename: str) -> GaussianData:
    """Read a binary_little_endian PLY file written by write_gaussian_ply."""
    with open(filename, "rb") as f:
        num_points = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                num_points = int(line.split()[-1])
            if line == "end_header":
                break

        raw = np.frombuffer(f.read(num_points * 14 * 4), dtype=np.float32).reshape(
            num_points, 14
        )

    return GaussianData(raw)
