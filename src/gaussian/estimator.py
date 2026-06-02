from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from .data import GaussianData
from .utils import agent_to_colmap, opacity_to_logit, rgb_to_sh


class GaussianEstimator:
    """Generates 3D Gaussian Splatting attributes from colored point clouds.

    Uses a two-stage voxel aggregation pipeline:
      1. Point-to-voxel: aggregate points into voxel statistics
      2. Voxel-to-neighborhood: merge 27-neighborhood statistics

    Optionally merges similar Gaussians via KD-tree neighbor search and
    union-find connected components.

    Args:
        voxel_size: Spatial resolution for voxelization (meters).
        min_points_per_voxel: Minimum points in a neighborhood to produce a Gaussian.
        covariance_regularization: Small diagonal added to covariance for stability.
        init_opacity: Initial opacity value [0, 1].
        merge_enabled: Whether to merge similar Gaussians after estimation.
        merge_lambda_sigma: Weight for covariance distance in merge metric.
        merge_lambda_c: Weight for color distance in merge metric.
        merge_tau: Curvature factor for adaptive merge threshold.
        merge_neighbor_dist: Search radius for merge neighbors (default: voxel_size).
    """

    def __init__(
        self,
        voxel_size: float = 0.015,
        min_points_per_voxel: int = 5,
        covariance_regularization: float = 1e-6,
        init_opacity: float = 1.0,
        merge_enabled: bool = False,
        merge_lambda_sigma: float = 0.6,
        merge_lambda_c: float = 0.4,
        merge_tau: float = -20.0,
        merge_neighbor_dist: Optional[float] = None,
    ):
        self.voxel_size = voxel_size
        self.min_points_per_voxel = min_points_per_voxel
        self.covariance_regularization = covariance_regularization
        self.init_opacity = init_opacity
        self.merge_enabled = merge_enabled
        self.merge_lambda_sigma = merge_lambda_sigma
        self.merge_lambda_c = merge_lambda_c
        self.merge_tau = merge_tau
        self.merge_neighbor_dist = (
            merge_neighbor_dist if merge_neighbor_dist is not None else voxel_size
        )

    def estimate(
        self,
        positions: np.ndarray,
        rgb_colors: np.ndarray,
        merge: Optional[bool] = None,
    ) -> GaussianData:
        """Generate 3DGS attributes from a colored point cloud.

        Input positions should be in Agent (xFyLzU) coordinates.
        Returns GaussianData in COLMAP (xRyDzF) coordinates.

        Args:
            positions: (N, 3) float array of 3D point coordinates.
            rgb_colors: (N, 3) float array of RGB colors in [0, 1].
            merge: Override ``merge_enabled`` for this call.

        Returns:
            GaussianData in COLMAP coordinates.
        """
        should_merge = self.merge_enabled if merge is None else merge
        intermediate = self._generate(positions, rgb_colors)

        if not intermediate:
            return GaussianData(np.empty((0, GaussianData._STRIDE), np.float32))

        if should_merge:
            intermediate = self._merge(intermediate)

        # Convert RGB to SH DC coefficients
        intermediate["f_dc"] = rgb_to_sh(intermediate.pop("rgb"))
        for key in ("sigmas", "kappas", "counts"):
            intermediate.pop(key, None)

        # Agent → COLMAP
        intermediate["means"], intermediate["rotations"] = agent_to_colmap(
            intermediate["means"], intermediate["rotations"]
        )

        return GaussianData.from_dict(intermediate)

    # ------------------------------------------------------------------
    # Internal pipeline (operates in Agent coordinates)
    # ------------------------------------------------------------------

    def _generate(self, positions: np.ndarray, rgb_colors: np.ndarray) -> dict:
        N = positions.shape[0]

        # Pre-compute outer products (N, 9)
        points_outer = (positions[:, :, None] * positions[:, None, :]).reshape(N, 9)

        # Quantize coordinates
        grid_coords = np.floor(positions / self.voxel_size).astype(np.int64)

        # Spatial hash
        min_coords = grid_coords.min(axis=0)
        shifted = grid_coords - min_coords
        mx, my, _ = shifted.max(axis=0) + 1
        voxel_keys = shifted[:, 0] + shifted[:, 1] * mx + shifted[:, 2] * mx * my

        # Sort & group
        sort_idx = np.argsort(voxel_keys)
        sorted_keys = voxel_keys[sort_idx]
        boundary = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
        reduce_at = np.nonzero(boundary)[0]

        unique_keys = sorted_keys[reduce_at]
        # Decode coordinates for neighborhood search
        uz = unique_keys // (mx * my)
        rem = unique_keys % (mx * my)
        uy = rem // mx
        ux = rem % mx
        unique_coords = np.vstack((ux, uy, uz)).T + min_coords

        # Aggregate base statistics
        sorted_pos = positions[sort_idx]
        sorted_rgb = rgb_colors[sort_idx]
        sorted_outer = points_outer[sort_idx]

        base_sum_pos = np.add.reduceat(sorted_pos, reduce_at, axis=0)
        base_sum_rgb = np.add.reduceat(sorted_rgb, reduce_at, axis=0)
        base_sum_outer = np.add.reduceat(sorted_outer, reduce_at, axis=0)
        base_counts = np.diff(np.concatenate((reduce_at, [N])))

        # --- Stage 2: 27-neighborhood aggregation ---
        M = len(base_counts)
        offsets = np.array(
            np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype=np.int64
        ).T.reshape(-1, 3)

        expanded_coords = (unique_coords[:, None, :] + offsets[None, :, :]).reshape(
            -1, 3
        )
        exp_sum_pos = np.repeat(base_sum_pos, 27, axis=0)
        exp_sum_rgb = np.repeat(base_sum_rgb, 27, axis=0)
        exp_sum_outer = np.repeat(base_sum_outer, 27, axis=0)
        exp_counts = np.repeat(base_counts, 27)

        # Re-hash expanded coordinates
        min_c2 = expanded_coords.min(axis=0)
        shifted2 = expanded_coords - min_c2
        mx2, my2, _ = shifted2.max(axis=0) + 1
        keys2 = shifted2[:, 0] + shifted2[:, 1] * mx2 + shifted2[:, 2] * mx2 * my2

        sort_idx2 = np.argsort(keys2)
        sorted_keys2 = keys2[sort_idx2]
        boundary2 = np.concatenate(([True], sorted_keys2[1:] != sorted_keys2[:-1]))
        reduce_at2 = np.nonzero(boundary2)[0]

        final_sum_pos = np.add.reduceat(exp_sum_pos[sort_idx2], reduce_at2, axis=0)
        final_sum_rgb = np.add.reduceat(exp_sum_rgb[sort_idx2], reduce_at2, axis=0)
        final_sum_outer = np.add.reduceat(exp_sum_outer[sort_idx2], reduce_at2, axis=0)
        final_counts = np.add.reduceat(exp_counts[sort_idx2], reduce_at2)

        # Filter sparse voxels
        valid = final_counts >= self.min_points_per_voxel
        if not np.any(valid):
            return {}

        final_sum_pos = final_sum_pos[valid]
        final_sum_rgb = final_sum_rgb[valid]
        final_sum_outer = final_sum_outer[valid]
        final_counts = final_counts[valid]

        # Parameter estimation
        means = final_sum_pos / final_counts[:, None]
        rgb = final_sum_rgb / final_counts[:, None]

        mean_outer = final_sum_outer / final_counts[:, None]
        mu_muT = (means[:, :, None] * means[:, None, :]).reshape(-1, 9)
        sigmas = (mean_outer - mu_muT).reshape(-1, 3, 3)

        # Regularization
        sigmas += self.covariance_regularization * np.eye(3)

        # Curvature
        evals = np.linalg.eigvalsh(sigmas)
        evals = np.maximum(evals, 1e-9)
        kappas = np.min(evals, axis=1) / np.sum(evals, axis=1)

        # Decompose covariance to scale + rotation
        scales, rots = _decompose_covariance_batch(sigmas)
        opacities = np.full(
            len(means), opacity_to_logit(self.init_opacity), dtype=np.float32
        )

        return {
            "means": means.astype(np.float32),
            "rgb": rgb.astype(np.float32),
            "opacities": opacities,
            "scales": scales,
            "rotations": rots,
            "sigmas": sigmas.astype(np.float32),
            "kappas": kappas.astype(np.float32),
            "counts": final_counts.astype(np.float32),
        }

    # -- Instance-level merge (operates on GaussianData objects) --

    def merge_gaussian_data(
        self, gauss_a: GaussianData, gauss_b: GaussianData
    ) -> GaussianData:
        """Merge two GaussianData with voxel-based deduplication.

        Gaussians are assigned to voxels (using ``means`` and ``self.voxel_size``).
        Voxels unique to A or B are kept as-is; overlapping voxels are averaged
        with equal weight across all 14 attributes.
        """
        na, nb = gauss_a.num_gaussians, gauss_b.num_gaussians
        if na == 0:
            return gauss_b
        if nb == 0:
            return gauss_a

        # Compute voxel keys with a shared coordinate frame
        all_means = np.concatenate([gauss_a.means, gauss_b.means], axis=0)
        grid = np.floor(all_means / self.voxel_size).astype(np.int64)
        min_c = grid.min(axis=0)
        shifted = grid - min_c
        mx, my, _ = shifted.max(axis=0) + 1
        all_keys = shifted[:, 0] + shifted[:, 1] * mx + shifted[:, 2] * mx * my

        keys_a = all_keys[:na]
        keys_b = all_keys[na:]

        # Tag origin: 0 = A-only, 1 = B-only, 2 = overlap
        all_keys = np.concatenate([keys_a, keys_b])
        origin = np.concatenate([np.zeros(na, np.int8), np.ones(nb, np.int8)])

        sort_idx = np.argsort(all_keys)
        sorted_keys = all_keys[sort_idx]
        sorted_origin = origin[sort_idx]

        boundary = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
        reduce_at = np.nonzero(boundary)[0]
        group_sizes = np.diff(np.concatenate((reduce_at, [len(sorted_keys)])))

        # Determine per-group type: 0=A-only, 1=B-only, 2=overlap
        has_a = np.add.reduceat((sorted_origin == 0).astype(np.int32), reduce_at) > 0
        has_b = np.add.reduceat((sorted_origin == 1).astype(np.int32), reduce_at) > 0
        group_type = has_a.astype(np.int8) + has_b.astype(
            np.int8
        )  # 0=impossible, 1=single, 2=overlap

        # Data in sorted order
        sorted_data = np.concatenate([gauss_a._data, gauss_b._data])[sort_idx]

        # Overlapping groups: equal-weight average per group
        overlap_mask = group_type == 2
        overlap_starts = reduce_at[overlap_mask]
        overlap_sizes = group_sizes[overlap_mask]
        overlap_ends = overlap_starts + overlap_sizes

        if len(overlap_starts) == 0:
            return GaussianData(sorted_data[reduce_at[group_type == 1]])

        overlap_avg = np.stack(
            [
                sorted_data[s:e].mean(axis=0)
                for s, e in zip(overlap_starts, overlap_ends)
            ]
        )

        # Non-overlapping groups: keep first row (same voxel key → just pick one)
        single_mask = group_type == 1
        single_data = sorted_data[reduce_at[single_mask]]

        if len(overlap_avg) == 0:
            return GaussianData(single_data)
        if len(single_data) == 0:
            return GaussianData(overlap_avg)

        return GaussianData(np.concatenate([single_data, overlap_avg], axis=0))

    # -- Estimation-time downsampling (operates on intermediate dicts) --

    def _merge(self, data: dict) -> dict:
        """Merge similar Gaussians using KD-tree neighbor search."""
        sigmas = data.get("sigmas")
        kappas = data.get("kappas")
        counts = data.get("counts")

        if sigmas is None or kappas is None or counts is None:
            return data

        means = data["means"]
        rgb = data["rgb"]
        N = len(means)

        tree = cKDTree(means)
        pairs = tree.query_pairs(r=self.merge_neighbor_dist, output_type="ndarray")

        if len(pairs) == 0:
            return data

        i_idx, j_idx = pairs[:, 0], pairs[:, 1]

        # Distance metrics
        dist_mu = np.linalg.norm(means[i_idx] - means[j_idx], axis=1)
        dist_rgb = np.linalg.norm(rgb[i_idx] - rgb[j_idx], axis=1)
        diff_sigma = sigmas[i_idx] - sigmas[j_idx]
        dist_sigma = np.sqrt(np.sum(diff_sigma**2, axis=(1, 2)))

        D = (
            dist_mu
            + self.merge_lambda_sigma * dist_sigma
            + self.merge_lambda_c * dist_rgb
        )

        # Adaptive threshold
        sum_k = kappas[i_idx] + kappas[j_idx]
        threshold = 2.0 * self.voxel_size * (1.0 + self.merge_tau * sum_k)
        merge_mask = D < threshold
        valid_pairs = pairs[merge_mask]

        if len(valid_pairs) == 0:
            return data

        # Union-Find via connected components
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        rows, cols = valid_pairs[:, 0], valid_pairs[:, 1]
        graph = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(N, N))
        n_components, labels = connected_components(graph, directed=False)

        # Aggregate parameters with bincount
        weights = counts
        new_counts = np.bincount(labels, weights=weights)

        w_means = means * weights[:, None]
        w_rgb = rgb * weights[:, None]

        new_means = np.vstack(
            [np.bincount(labels, weights=w_means[:, d]) for d in range(3)]
        ).T / np.maximum(new_counts[:, None], 1.0)
        new_rgb = np.vstack(
            [np.bincount(labels, weights=w_rgb[:, d]) for d in range(3)]
        ).T / np.maximum(new_counts[:, None], 1.0)

        # Correct sigma merging via second-moment aggregation
        mu_muT = means[:, :, None] * means[:, None, :]
        second_moment = (sigmas + mu_muT).reshape(N, 9) * weights[:, None]

        sum_sm = np.zeros((n_components, 9), dtype=np.float64)
        for k in range(9):
            sum_sm[:, k] = np.bincount(labels, weights=second_moment[:, k])

        expected_xxT = sum_sm.reshape(n_components, 3, 3) / np.maximum(
            new_counts[:, None, None], 1.0
        )
        new_mu_muT = new_means[:, :, None] * new_means[:, None, :]
        new_sigmas = expected_xxT - new_mu_muT
        new_sigmas += 1e-6 * np.eye(3)

        new_scales, new_rots = _decompose_covariance_batch(new_sigmas)
        new_opacities = np.full(
            n_components, opacity_to_logit(self.init_opacity), dtype=np.float32
        )

        # Curvature for merged result
        new_evals = np.linalg.eigvalsh(new_sigmas)
        new_evals = np.maximum(new_evals, 1e-9)
        new_kappas = np.min(new_evals, axis=1) / np.sum(new_evals, axis=1)

        return {
            "means": new_means.astype(np.float32),
            "rgb": new_rgb.astype(np.float32),
            "opacities": new_opacities,
            "scales": new_scales,
            "rotations": new_rots,
            "sigmas": new_sigmas.astype(np.float32),
            "kappas": new_kappas.astype(np.float32),
            "counts": new_counts.astype(np.float32),
        }


def _decompose_covariance_batch(
    covariances: np.ndarray,
) -> tuple:
    """Decompose covariance matrices to log-scale and quaternion rotation.

    Args:
        covariances: (N, 3, 3) covariance matrices.

    Returns:
        (scales, rotations) where scales is (N, 3) log-space and
        rotations is (N, 4) quaternions in (w, x, y, z) order.
    """
    evals, evecs = np.linalg.eigh(covariances)
    evals = np.maximum(evals, 1e-9)
    scales = np.log(np.sqrt(evals)).astype(np.float32)

    # Ensure proper rotation (det = +1) by flipping sign of one column
    dets = np.linalg.det(evecs)
    evecs[dets < 0, :, -1] *= -1

    quats = R.from_matrix(evecs).as_quat()  # (x, y, z, w)
    rotations = np.roll(quats, 1, axis=1).astype(np.float32)  # -> (w, x, y, z)

    return scales, rotations
