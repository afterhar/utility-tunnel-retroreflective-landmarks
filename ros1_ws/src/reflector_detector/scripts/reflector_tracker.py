#!/usr/bin/env python3
"""Multi-frame temporal tracker using mature open-source algorithms.

- Kalman filter (filterpy): per-track state prediction
- Hungarian algorithm (scipy): optimal track-detection association
- Confidence EMA + 4-state machine: adapted from V1 C++ tracker
"""
import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Reuse ClusterBounds from sparse_detector
try:
    from reflector_sparse_detector import ClusterBounds
except ImportError:
    @dataclass
    class ClusterBounds:
        center_x: float = 0.0; center_y: float = 0.0; center_z: float = 0.0
        x_min: float = 0.0; x_max: float = 0.0
        y_min: float = 0.0; y_max: float = 0.0
        z_min: float = 0.0; z_max: float = 0.0
        x_range: float = 0.0; y_range: float = 0.0; z_range: float = 0.0
        xy_range: float = 0.0
        mean_intensity: float = 0.0; std_xy: float = 0.0
        z_band_count: int = 0; point_count: int = 0
        score_intensity: float = 0.0; score_shape: float = 0.0
        source_mask: int = 0
        accepted_dense: bool = False; accepted_sparse: bool = False
        near_sparse_rescue: bool = False
        aspect_ratio: float = 1.0
        background_intensity: float = 0.0


class ObservationState(IntEnum):
    TENTATIVE = 0; STABLE = 1; DEGRADED = 2; LOST = 3

SOURCE_DENSE = 1; SOURCE_SPARSE = 2


@dataclass
class TrackerParams:
    match_distance_threshold: float = 0.35
    max_missing_frames: int = 6
    # Dense state machine
    dense_stable_grace_frames: int = 3
    dense_degrade_window: int = 5
    dense_degrade_max_hits: int = 1
    dense_degrade_min_confidence: float = 0.56
    dense_recover_window: int = 3
    dense_recover_min_hits: int = 2
    dense_recover_min_confidence: float = 0.64
    # Sparse state machine
    sparse_stable_window: int = 5
    sparse_stable_min_hits: int = 2
    sparse_stable_min_confidence: float = 0.54
    sparse_degrade_window: int = 7
    sparse_degrade_max_hits: int = 1
    sparse_degrade_min_confidence: float = 0.44
    sparse_recover_window: int = 4
    sparse_recover_min_hits: int = 2
    sparse_recover_min_confidence: float = 0.60
    sparse_stable_grace_frames: int = 4
    # Output hold (disabled by default - causes position lag)
    enable_output_hold: bool = False
    output_hold_frames: int = 1
    output_hold_max_missing: int = 2
    # Rotation
    adaptive_rotation_tracking: bool = True
    rotation_rate_threshold: float = 0.35
    rotation_gate_expand: float = 1.35
    rotation_missing_bonus: int = 2
    rotation_rebind_relax: float = 0.25
    # IMU weak prediction
    enable_imu_weak_prediction: bool = True
    weak_prediction_velocity_gain: float = 0.6
    weak_prediction_gate_expand: float = 1.35
    # Assignment
    global_assignment_max_candidates: int = 14
    # Kalman filter
    kf_process_noise: float = 0.5
    kf_measurement_noise: float = 0.1


@dataclass
class TrackedCluster:
    id: int = -1
    cluster: Optional[ClusterBounds] = None
    kf: Optional[KalmanFilter] = None           # Kalman filter for this track
    stable_count: int = 0
    missing_count: int = 0
    candidate_hits_in_window: int = 0
    frames_since_birth: int = 0
    confidence_raw: float = 0.0
    confidence_ema: float = 0.0
    state: int = ObservationState.TENTATIVE
    gate_radius: float = 0.5
    source_mask: int = 0
    near_sparse_rescue: bool = False
    stable_grace_remaining: int = 0
    recent_hits: deque = field(default_factory=deque)
    rotation_adapted: bool = False
    last_score_intensity: float = 0.0
    last_score_shape: float = 0.0
    last_score_temporal: float = 0.0
    last_score_prediction: float = 0.5
    last_association_cost: float = 1.0


@dataclass
class TrackedOutput:
    tracker_id: int = -1
    center_x: float = 0.0; center_y: float = 0.0; center_z: float = 0.0
    confidence: float = 0.0
    state: int = ObservationState.TENTATIVE
    stable_count: int = 0; missing_count: int = 0
    gate_radius: float = 0.0
    score_intensity: float = 0.0; score_shape: float = 0.0
    score_temporal: float = 0.0; score_prediction: float = 0.5
    association_cost: float = 0.0
    source_mask: int = 0
    publish_eligible: bool = False
    matched_by_prediction: bool = False
    rotation_adapted: bool = False
    prediction_distance: float = 0.0
    rotation_rate: float = 0.0


@dataclass
class OdometryState:
    valid: bool = False; prev_valid: bool = False
    T_curr: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_prev: np.ndarray = field(default_factory=lambda: np.eye(4))
    delta_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance_xy_trace: float = 0.0


@dataclass
class ImuState:
    valid: bool = False
    delta_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    dt: float = 0.0
    angular_velocity: float = 0.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _make_kf(dt: float, process_noise: float, meas_noise: float) -> KalmanFilter:
    """Create a 6-state constant-velocity Kalman filter: [x, y, z, vx, vy, vz]."""
    kf = KalmanFilter(dim_x=6, dim_z=3)
    # State transition: constant velocity
    kf.F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1],
    ])
    # Measurement: observe [x, y, z]
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ])
    # Process noise
    q = process_noise
    kf.Q = np.eye(6) * q
    kf.Q[3:, 3:] *= 0.5  # less noise on velocity
    # Measurement noise
    kf.R = np.eye(3) * meas_noise
    # Initial covariance
    kf.P = np.eye(6) * 0.5
    kf.P[3:, 3:] *= 0.2
    return kf


class Tracker:
    """Multi-frame temporal tracker using Kalman filter + Hungarian assignment."""

    def __init__(self, params: TrackerParams):
        self.p = params
        self._tracks: List[TrackedCluster] = []
        self._next_id: int = 1
        self._last_stamp: Optional[float] = None
        self._dt: float = 0.1  # estimated frame interval

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._last_stamp = None

    # ── Main entry point ──────────────────────────────────────────
    def update(
        self,
        clusters: List[ClusterBounds],
        odom: OdometryState,
        imu: ImuState,
        stamp_sec: float,
    ) -> List[TrackedOutput]:
        # Compute real dt
        if self._last_stamp is not None and stamp_sec > self._last_stamp:
            self._dt = min(stamp_sec - self._last_stamp, 1.0)
        self._last_stamp = stamp_sec
        dt = self._dt

        high_rotation = self._is_high_rotation(imu)

        # Update Kalman filter transition matrix with current dt
        for t in self._tracks:
            if t.kf is not None:
                t.kf.F[0, 3] = dt
                t.kf.F[1, 4] = dt
                t.kf.F[2, 5] = dt

        # 1. Predict all tracks via Kalman filter
        predictions = []
        for t in self._tracks:
            if t.kf is not None:
                t.kf.predict()
                predictions.append(t.kf.x[:3, 0].copy())
            else:
                predictions.append(None)
            t.frames_since_birth += 1
            t.rotation_adapted = False
            if high_rotation and t.state in (ObservationState.STABLE, ObservationState.DEGRADED):
                t.gate_radius = 0.5 * self.p.rotation_gate_expand
                t.rotation_adapted = True
            else:
                t.gate_radius = 0.5

        # 2. Build cost matrix: track × cluster
        M = len(self._tracks)
        N = len(clusters)
        if M == 0 and N == 0:
            return []
        if M == 0:
            # All new tracks
            for ci in range(N):
                self._create_track(clusters[ci], dt)
            return []
        if N == 0:
            # All tracks miss
            for t in self._tracks:
                self._handle_miss(t, high_rotation)
            self._prune_lost()
            return self._build_outputs()

        cost = np.full((M, N), 1e6, dtype=np.float64)
        for ti in range(M):
            pred = predictions[ti]
            track = self._tracks[ti]
            if pred is None:
                continue
            for ci in range(N):
                cl = clusters[ci]
                obs = np.array([cl.center_x, cl.center_y, cl.center_z])
                dist = float(np.linalg.norm(obs - pred))
                cost[ti, ci] = dist

        # 3. Hungarian assignment
        track_idx, cluster_idx = linear_sum_assignment(cost)
        matched_tracks = set()
        matched_clusters = set()

        for ti, ci in zip(track_idx, cluster_idx):
            if cost[ti, ci] > self.p.match_distance_threshold * 3.0:
                continue  # too far, treat as unmatched
            matched_tracks.add(ti)
            matched_clusters.add(ci)
            track = self._tracks[ti]
            cl = clusters[ci]

            # Kalman update
            obs = np.array([cl.center_x, cl.center_y, cl.center_z])
            if track.kf is not None:
                track.kf.update(obs)

            track.cluster = deepcopy(cl)
            track.source_mask = cl.source_mask
            track.near_sparse_rescue = getattr(cl, 'near_sparse_rescue', False)
            track.stable_count += 1
            track.missing_count = 0
            track.candidate_hits_in_window = min(track.candidate_hits_in_window + 1, 20)
            track.recent_hits.append(True)
            if len(track.recent_hits) > 30:
                track.recent_hits.popleft()

            # Confidence EMA
            si = getattr(cl, 'score_intensity', 0.5)
            ss = getattr(cl, 'score_shape', 0.5)
            track.last_score_intensity = si
            track.last_score_shape = ss
            if track.stable_count <= 1:
                track.confidence_raw = 0.35*si + 0.25*ss + 0.20*0.5 + 0.20*0.5
                track.confidence_ema = track.confidence_raw
            else:
                track.confidence_raw = 0.35*si + 0.25*ss + 0.20*min(track.stable_count,5)/5.0 + 0.20*0.5
                track.confidence_ema = _clamp(0.4*track.confidence_raw + 0.6*track.confidence_ema, 0.0, 1.0)
            track.last_association_cost = cost[ti, ci]

            self._update_state(track, matched=True, high_rotation=high_rotation)

        # 4. Handle unmatched tracks
        for ti in range(M):
            if ti in matched_tracks:
                continue
            self._handle_miss(self._tracks[ti], high_rotation)

        # 5. Create new tracks for unmatched clusters
        for ci in range(N):
            if ci in matched_clusters:
                continue
            self._create_track(clusters[ci], dt)

        # 6. Prune and build outputs
        self._prune_lost()
        return self._build_outputs()

    def _create_track(self, cluster: ClusterBounds, dt: float):
        kf = _make_kf(dt, self.p.kf_process_noise, self.p.kf_measurement_noise)
        obs = np.array([cluster.center_x, cluster.center_y, cluster.center_z])
        kf.x[:3, 0] = obs
        kf.x[3:, 0] = 0.0  # initial velocity = 0

        si = getattr(cluster, 'score_intensity', 0.5)
        ss = getattr(cluster, 'score_shape', 0.5)

        track = TrackedCluster(
            id=self._next_id,
            cluster=deepcopy(cluster),
            kf=kf,
            source_mask=cluster.source_mask,
            near_sparse_rescue=getattr(cluster, 'near_sparse_rescue', False),
            stable_count=1,
            frames_since_birth=1,
            confidence_raw=0.35*si + 0.25*ss + 0.20*0.2 + 0.20*0.5,
            gate_radius=0.5,
        )
        track.confidence_ema = track.confidence_raw
        track.recent_hits.append(True)
        self._tracks.append(track)
        self._next_id += 1

    def _handle_miss(self, track: TrackedCluster, high_rotation: bool):
        track.missing_count += 1
        track.candidate_hits_in_window = max(0, track.candidate_hits_in_window - 1)
        track.recent_hits.append(False)
        if len(track.recent_hits) > 30:
            track.recent_hits.popleft()
        track.confidence_ema *= 0.92
        self._update_state(track, matched=False, high_rotation=high_rotation)

    # ── State machine ──────────────────────────────────────────────
    def _update_state(self, track, matched, high_rotation):
        was_state = track.state
        bonus = self.p.rotation_missing_bonus if high_rotation else 0
        sparse_only = (track.source_mask & SOURCE_DENSE) == 0

        if track.state == ObservationState.TENTATIVE:
            if matched:
                self._try_promote(track, sparse_only)
            elif track.missing_count > 6:
                track.state = ObservationState.LOST
        elif track.state == ObservationState.STABLE:
            if track.stable_grace_remaining > 0 and track.missing_count == 0:
                track.stable_grace_remaining -= 1
            elif track.stable_grace_remaining <= 0:
                self._check_degrade(track, sparse_only, high_rotation)
        elif track.state == ObservationState.DEGRADED:
            eff_max = self.p.max_missing_frames + bonus
            if track.missing_count > eff_max:
                track.state = ObservationState.LOST
            elif matched:
                track.state = ObservationState.STABLE  # instant recovery
                track.stable_grace_remaining = (self.p.sparse_stable_grace_frames
                                                if sparse_only
                                                else self.p.dense_stable_grace_frames)

    def _try_promote(self, track, sparse_only):
        if sparse_only:
            hits = self._count_hits(track, self.p.sparse_stable_window)
            if hits >= self.p.sparse_stable_min_hits and track.confidence_ema >= self.p.sparse_stable_min_confidence:
                track.state = ObservationState.STABLE
                track.stable_grace_remaining = self.p.sparse_stable_grace_frames
        else:
            if track.stable_count >= 3 and track.confidence_ema >= 0.55 and track.candidate_hits_in_window >= 2:
                track.state = ObservationState.STABLE
                track.stable_grace_remaining = self.p.dense_stable_grace_frames

    def _check_degrade(self, track, sparse_only, high_rotation):
        bonus = self.p.rotation_missing_bonus if high_rotation else 0
        if sparse_only:
            w = self.p.sparse_degrade_window
            hits = self._count_hits(track, w)
            if hits <= self.p.sparse_degrade_max_hits + bonus or track.confidence_ema < self.p.sparse_degrade_min_confidence:
                track.state = ObservationState.DEGRADED
        else:
            if track.missing_count >= 3 + bonus:
                track.state = ObservationState.DEGRADED

    # ── Helpers ────────────────────────────────────────────────────
    def _count_hits(self, track, window):
        hits_list = list(track.recent_hits)
        n = len(hits_list)
        w = min(window, n)
        return sum(1 for i in range(n - w, n) if hits_list[i])

    def _is_high_rotation(self, imu):
        return (self.p.adaptive_rotation_tracking and imu.valid and
                imu.angular_velocity >= self.p.rotation_rate_threshold)

    def _prune_lost(self):
        self._tracks = [t for t in self._tracks if t.state != ObservationState.LOST]

    def _build_outputs(self) -> List[TrackedOutput]:
        outputs = []
        for t in self._tracks:
            publish_eligible = (
                t.state in (ObservationState.STABLE, ObservationState.DEGRADED)
                and t.missing_count == 0
            )
            out = TrackedOutput(
                tracker_id=t.id,
                confidence=t.confidence_ema,
                state=t.state,
                stable_count=t.stable_count,
                missing_count=t.missing_count,
                gate_radius=t.gate_radius,
                score_intensity=t.last_score_intensity,
                score_shape=t.last_score_shape,
                score_temporal=t.last_score_temporal,
                score_prediction=t.last_score_prediction,
                association_cost=t.last_association_cost,
                source_mask=t.source_mask,
                publish_eligible=publish_eligible,
                rotation_adapted=t.rotation_adapted,
            )
            if t.cluster is not None:
                out.center_x = t.cluster.center_x
                out.center_y = t.cluster.center_y
                out.center_z = t.cluster.center_z
            elif t.kf is not None:
                out.center_x = float(t.kf.x[0, 0])
                out.center_y = float(t.kf.x[1, 0])
                out.center_z = float(t.kf.x[2, 0])
            else:
                out.publish_eligible = False
            outputs.append(out)
        return outputs
