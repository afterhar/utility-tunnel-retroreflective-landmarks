#!/usr/bin/env python3
"""Sparse column detector for V3 reflector detection.
Finds reflector columns with very few LiDAR returns (2-4 points per scan)
by XY-radius aggregation and relaxed shape checks.

Ported from V1 C++ evaluateSparseCandidate / buildSparseClusterBounds.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np


@dataclass
class SparseParams:
    enable: bool = True
    xy_radius: float = 0.09
    min_range: float = 2.0
    min_points: int = 2
    min_z_range: float = 0.12
    max_xy_range: float = 0.18
    max_xy_std: float = 0.055
    min_z_bands: int = 2
    min_mean_intensity: float = 108.0
    # Near-sparse rescue
    near_sparse_enable: bool = True
    near_sparse_min_range: float = 1.2
    near_sparse_max_range: float = 2.5
    near_sparse_min_points: int = 2
    near_sparse_min_z_range: float = 0.14
    near_sparse_max_xy_range: float = 0.16
    near_sparse_max_xy_std: float = 0.035
    near_sparse_min_z_bands: int = 3
    near_sparse_min_mean_intensity: float = 108.0
    near_sparse_min_contrast_ratio: float = 1.10
    z_band_step: float = 0.04
    min_aspect_ratio: float = 1.0


@dataclass
class ClusterBounds:
    """Per-frame cluster produced by frontend."""
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    x_range: float = 0.0
    y_range: float = 0.0
    z_range: float = 0.0
    xy_range: float = 0.0
    mean_intensity: float = 0.0
    background_intensity: float = 0.0
    std_xy: float = 0.0
    z_band_count: int = 0
    point_count: int = 0
    score_intensity: float = 0.0
    score_shape: float = 0.0
    # source: 1=dense, 2=sparse, 3=both
    source_mask: int = 0
    accepted_dense: bool = False
    accepted_sparse: bool = False
    near_sparse_rescue: bool = False
    aspect_ratio: float = 1.0


class SparseDetector:
    """Detect sparse reflector columns from intensity-filtered points.

    Operates on strong points that may not have been clustered by dense XY DBSCAN.
    Uses XY-radius BFS aggregation with relaxed shape constraints for far-range,
    and stricter near-sparse constraints for close-range rescue.
    """

    def __init__(self, params: SparseParams):
        self.p = params

    def detect(
        self,
        strong_points: np.ndarray,          # (N, 4+): [x, y, z, intensity, ...]
        used_indices: Set[int],             # already claimed by dense clustering
    ) -> List[ClusterBounds]:
        """Return sparse ClusterBounds candidates from unclaimed strong points.

        Uses cKDTree for O(n log n) neighbor queries when available,
        falling back to grid-based BFS otherwise.
        """
        if not self.p.enable or len(strong_points) == 0:
            return []

        # Filter to unclaimed points
        n_total = len(strong_points)
        unclaimed_mask = np.ones(n_total, dtype=bool)
        if used_indices:
            for idx in used_indices:
                if idx < n_total:
                    unclaimed_mask[idx] = False

        unclaimed_idx = np.where(unclaimed_mask)[0]
        if len(unclaimed_idx) < self.p.min_points:
            return []

        unclaimed_pts = strong_points[unclaimed_idx]

        # Use cKDTree for fast radius search
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(unclaimed_pts[:, :2])
            radius = self.p.xy_radius
            nbr_lists = tree.query_ball_point(unclaimed_pts[:, :2], radius)

            # DBSCAN-style flood fill
            labels = np.full(len(unclaimed_pts), -2, dtype=np.int32)  # -2=unvisited
            cluster_id = 0
            clusters: dict = {}

            for i in range(len(unclaimed_pts)):
                if labels[i] != -2:
                    continue
                nbrs = nbr_lists[i]
                if len(nbrs) < self.p.min_points:
                    labels[i] = -1  # noise
                    continue
                labels[i] = cluster_id
                clusters[cluster_id] = [i]
                queue = list(nbrs)
                queued = set(nbrs)
                queued.discard(i)
                while queue:
                    cur = queue.pop()
                    if labels[cur] == -1:
                        labels[cur] = cluster_id
                        clusters[cluster_id].append(cur)
                    if labels[cur] != -2:
                        continue
                    labels[cur] = cluster_id
                    clusters[cluster_id].append(cur)
                    if len(nbr_lists[cur]) >= self.p.min_points:
                        for nb in nbr_lists[cur]:
                            if nb not in queued and labels[nb] == -2:
                                queued.add(nb)
                                queue.append(nb)
                cluster_id += 1

            # Convert clusters to candidates
            candidates: List[ClusterBounds] = []
            for indices in clusters.values():
                # Map back to original strong_points indices
                orig_indices = [int(unclaimed_idx[i]) for i in indices]
                if len(orig_indices) < self.p.min_points:
                    continue
                bounds = self._build_bounds(strong_points, orig_indices)
                if bounds is None:
                    continue
                r = math.sqrt(bounds.center_x**2 + bounds.center_y**2 + bounds.center_z**2)
                if r >= self.p.min_range:
                    if self._evaluate_far_sparse(bounds):
                        bounds.accepted_sparse = True
                        bounds.source_mask |= 2
                        candidates.append(bounds)
                elif self.p.near_sparse_enable and self.p.near_sparse_min_range <= r <= self.p.near_sparse_max_range:
                    if self._evaluate_near_sparse(bounds):
                        bounds.accepted_sparse = True
                        bounds.source_mask |= 2
                        bounds.near_sparse_rescue = True
                        candidates.append(bounds)
            return candidates

        except ImportError:
            pass

        # Fallback: grid-based BFS
        return self._detect_grid(strong_points, used_indices)

    def _detect_grid(
        self,
        strong_points: np.ndarray,
        used_indices: Set[int],
    ) -> List[ClusterBounds]:
        """Fallback grid-based sparse detection."""
        candidates: List[ClusterBounds] = []
        visited: Set[int] = set()
        xs = strong_points[:, 0]
        ys = strong_points[:, 1]
        cell_size = max(self.p.xy_radius, 0.01)
        grid: dict = {}
        for i in range(len(strong_points)):
            if i in used_indices:
                continue
            cx = int(xs[i] / cell_size)
            cy = int(ys[i] / cell_size)
            grid.setdefault((cx, cy), []).append(i)
        radius_sq = self.p.xy_radius * self.p.xy_radius

        def _region_query(idx):
            px, py = xs[idx], ys[idx]
            cx0, cy0 = int(px / cell_size), int(py / cell_size)
            nbrs = []
            for dcx in (-1, 0, 1):
                for dcy in (-1, 0, 1):
                    for cand in grid.get((cx0 + dcx, cy0 + dcy), ()):
                        if cand == idx:
                            continue
                        if (px - xs[cand])**2 + (py - ys[cand])**2 <= radius_sq:
                            nbrs.append(cand)
            return nbrs

        for i in range(len(strong_points)):
            if i in visited or i in used_indices:
                continue
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                cur = queue.pop()
                component.append(cur)
                for nb in _region_query(cur):
                    if nb not in visited and nb not in used_indices:
                        visited.add(nb)
                        queue.append(nb)
            if len(component) < self.p.min_points:
                continue
            bounds = self._build_bounds(strong_points, component)
            if bounds is None:
                continue
            r = math.sqrt(bounds.center_x**2 + bounds.center_y**2 + bounds.center_z**2)
            if r >= self.p.min_range:
                if self._evaluate_far_sparse(bounds):
                    bounds.accepted_sparse = True
                    bounds.source_mask |= 2
                    candidates.append(bounds)
            elif self.p.near_sparse_enable and self.p.near_sparse_min_range <= r <= self.p.near_sparse_max_range:
                if self._evaluate_near_sparse(bounds):
                    bounds.accepted_sparse = True
                    bounds.source_mask |= 2
                    bounds.near_sparse_rescue = True
                    candidates.append(bounds)
        return candidates

    def _build_bounds(
        self, cloud: np.ndarray, indices: List[int]
    ) -> Optional[ClusterBounds]:
        pts = cloud[np.asarray(indices, dtype=np.intp)]
        xyz = pts[:, :3]
        intensities = pts[:, 3]

        x_min, y_min, z_min = np.min(xyz, axis=0)
        x_max, y_max, z_max = np.max(xyz, axis=0)
        center = np.median(xyz, axis=0)

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        xy_range = max(x_range, y_range)
        var_xy = float(np.var(xyz[:, 0]) + np.var(xyz[:, 1]))
        std_xy = math.sqrt(max(0.0, var_xy))

        # Z band count
        z_step = max(self.p.z_band_step, 0.01)
        z_bins = np.floor((xyz[:, 2] - z_min) / z_step).astype(np.int32)
        z_band_count = int(np.unique(z_bins).size)

        aspect_ratio = z_range / max(xy_range, 0.01)
        mean_intensity = float(np.mean(intensities))

        return ClusterBounds(
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(0.5 * (z_min + z_max)),
            x_min=float(x_min), x_max=float(x_max),
            y_min=float(y_min), y_max=float(y_max),
            z_min=float(z_min), z_max=float(z_max),
            x_range=float(x_range), y_range=float(y_range),
            z_range=float(z_range),
            xy_range=float(xy_range),
            mean_intensity=mean_intensity,
            std_xy=std_xy,
            z_band_count=z_band_count,
            point_count=len(indices),
            aspect_ratio=float(aspect_ratio),
        )

    def _evaluate_far_sparse(self, b: ClusterBounds) -> bool:
        if b.point_count < self.p.min_points:
            return False
        if b.z_range < self.p.min_z_range:
            return False
        if b.xy_range > self.p.max_xy_range:
            return False
        if b.std_xy > self.p.max_xy_std:
            return False
        if b.z_band_count < self.p.min_z_bands:
            return False
        if b.mean_intensity < self.p.min_mean_intensity:
            return False
        if b.aspect_ratio < self.p.min_aspect_ratio:
            return False
        # Z-dominance check
        if b.z_range < b.x_range or b.z_range < b.y_range:
            return False
        return True

    def _evaluate_near_sparse(self, b: ClusterBounds) -> bool:
        if b.point_count < self.p.near_sparse_min_points:
            return False
        if b.z_range < self.p.near_sparse_min_z_range:
            return False
        if b.xy_range > self.p.near_sparse_max_xy_range:
            return False
        if b.std_xy > self.p.near_sparse_max_xy_std:
            return False
        if b.z_band_count < self.p.near_sparse_min_z_bands:
            return False
        if b.mean_intensity < self.p.near_sparse_min_mean_intensity:
            return False
        # Near-sparse requires higher aspect ratio (tall and thin)
        if b.aspect_ratio < 3.0:
            return False
        # Contrast check: cluster must be significantly brighter than background
        if self.p.near_sparse_min_contrast_ratio > 0.0 and b.background_intensity > 0.0:
            ratio = b.mean_intensity / max(b.background_intensity, 0.1)
            if ratio < self.p.near_sparse_min_contrast_ratio:
                return False
        return True
