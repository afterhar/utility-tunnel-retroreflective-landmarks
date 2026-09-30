#!/usr/bin/env python3
"""V3 Reflector Detector Node.

Combines V2's numpy-based per-frame detection pipeline with V1's multi-frame
temporal tracking, sparse column detection, and state machine.

Architecture:
  PointCloud -> numpy extract/filter -> XY DBSCAN cluster -> shape verify --+
                                                                            +-> fuse -> Tracker -> publish
  PointCloud -> Sparse column detect ---------------------------------------+
"""
import math
import os
import struct
import sys
import threading
import time

# Workspace devel path for message imports
_workspace_devel = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "devel", "lib", "python3", "dist-packages")
)
if os.path.isdir(_workspace_devel) and _workspace_devel not in sys.path:
    sys.path.insert(0, _workspace_devel)

import numpy as np
import rospy

try:
    from scipy.spatial import cKDTree as _scipy_cKDTree
except ImportError:
    _scipy_cKDTree = None

from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from reflector_detector.msg import ReflectorObservation, ReflectorObservationArray

# Local modules — add scripts source dir to path so catkin relay imports work
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from reflector_sparse_detector import SparseDetector, SparseParams, ClusterBounds
from imu_deskew import ImuDeskew
from reflector_tracker import (Tracker, TrackerParams, TrackedOutput,
                               OdometryState, ImuState,
                               ObservationState, SOURCE_DENSE, SOURCE_SPARSE)


# ── Clustering (V2-style XY DBSCAN) ──────────────────────────────

def _cluster_xy_grid(points, eps, min_samples):
    """Grid-accelerated DBSCAN in XY plane. Ported from V2."""
    count = len(points)
    if count == 0:
        return []
    xs = points[:, 0]
    ys = points[:, 1]
    eps_sq = eps * eps
    unvisited = -2
    noise = -1
    labels = [unvisited] * count
    cluster_id = 0
    cell_size = max(eps, 1e-6)
    grid = {}
    cell_xs = np.floor(xs / cell_size).astype(np.int32)
    cell_ys = np.floor(ys / cell_size).astype(np.int32)
    for idx in range(count):
        grid.setdefault((int(cell_xs[idx]), int(cell_ys[idx])), []).append(idx)

    def _region_query(idx):
        px, py = xs[idx], ys[idx]
        cx0, cy0 = int(cell_xs[idx]), int(cell_ys[idx])
        nbrs = []
        for dcx in (-1, 0, 1):
            for dcy in (-1, 0, 1):
                for cand in grid.get((cx0 + dcx, cy0 + dcy), ()):
                    dx = px - xs[cand]
                    dy = py - ys[cand]
                    if dx*dx + dy*dy <= eps_sq:
                        nbrs.append(cand)
        return nbrs

    clusters = {}
    for idx in range(count):
        if labels[idx] != unvisited:
            continue
        nbrs = _region_query(idx)
        if len(nbrs) < min_samples:
            labels[idx] = noise
            continue
        labels[idx] = cluster_id
        clusters[cluster_id] = [idx]
        queue = list(nbrs)
        queued = set(nbrs)
        queued.discard(idx)
        while queue:
            cur = queue.pop()
            if labels[cur] == noise:
                labels[cur] = cluster_id
                clusters[cluster_id].append(cur)
            if labels[cur] != unvisited:
                continue
            labels[cur] = cluster_id
            clusters[cluster_id].append(cur)
            cur_nbrs = _region_query(cur)
            if len(cur_nbrs) >= min_samples:
                for nb in cur_nbrs:
                    if nb not in queued and labels[nb] == unvisited:
                        queued.add(nb)
                        queue.append(nb)
        cluster_id += 1
    return list(clusters.values())


def _cluster_xy(points, eps, min_samples):
    """Dispatch clustering to grid or cKDTree."""
    if _scipy_cKDTree is not None:
        return _cluster_xy_ckdtree(points, eps, min_samples)
    return _cluster_xy_grid(points, eps, min_samples)


def _cluster_xy_ckdtree(points, eps, min_samples):
    """cKDTree-accelerated DBSCAN in XY."""
    count = len(points)
    if count == 0:
        return []
    tree = _scipy_cKDTree(points[:, :2])
    nbr_lists = tree.query_ball_point(points[:, :2], eps)
    core_mask = np.array([len(nl) >= min_samples for nl in nbr_lists], dtype=bool)
    labels = np.full(count, -2, dtype=np.int32)  # -2 = unvisited
    cluster_id = 0
    clusters = {}
    for idx in range(count):
        if labels[idx] != -2:
            continue
        if not core_mask[idx]:
            labels[idx] = -1
            continue
        labels[idx] = cluster_id
        clusters[cluster_id] = [idx]
        queue = list(nbr_lists[idx])
        queued = set(nbr_lists[idx])
        queued.discard(idx)
        while queue:
            cur = queue.pop()
            if labels[cur] == -1:
                labels[cur] = cluster_id
                clusters[cluster_id].append(cur)
            if labels[cur] != -2:
                continue
            labels[cur] = cluster_id
            clusters[cluster_id].append(cur)
            if core_mask[cur]:
                for nb in nbr_lists[cur]:
                    if nb not in queued and labels[nb] == -2:
                        queued.add(nb)
                        queue.append(nb)
        cluster_id += 1
    return list(clusters.values())


# ── Main V3 Node ──────────────────────────────────────────────────

class ReflectorDetectorV3:
    def __init__(self):
        self._load_params()
        self._init_state()
        self._setup_pubsub()

        # Start background worker thread (V2 pattern)
        self._worker = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker.start()
        rospy.loginfo("V3 Reflector Detector initialized (V2 frontend + V1 tracking)")

    def _load_params(self):
        """Load parameters with backward compatibility: v3_xxx -> xxx -> v2_xxx -> default."""
        def p(name, default):
            # Explicit/private params should beat launch defaults.  This lets
            # users override with `_intensity_threshold:=...` even when the
            # launch file also provides v2_* compatibility defaults.
            val = rospy.get_param("~v3_" + name, None)
            if val is not None:
                return val
            val = rospy.get_param("~_" + name, None)
            if val is not None:
                return val
            val = rospy.get_param("~" + name, None)
            if val is not None:
                return val
            val = rospy.get_param("~v2_" + name, None)
            if val is not None:
                return val
            return default

        self.input_topic = rospy.get_param("~input_topic", "/rslidar_points")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu/data")
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.frame_id_override = rospy.get_param("~frame_id", "")
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        self.publish_filtered_cloud = rospy.get_param("~publish_filtered_cloud", True)
        self.publish_processed_input_cloud = rospy.get_param("~publish_processed_input_cloud", True)
        self.debug_print_every_n = int(p("debug_print_every_n", 10))
        self.enable_timing_log = bool(p("enable_timing_log", True))

        # Frontend — intensity/filtering
        self.intensity_threshold = float(p("intensity_threshold", 80.0))
        self.z_min = float(p("z_min", -0.5))
        self.z_max = float(p("z_max", 0.6))
        self.min_detection_range = float(p("min_detection_range", 0.8))
        self.max_detection_range = float(p("max_detection_range", 20.0))
        self.cluster_voxel_size = float(p("cluster_voxel_size", 0.0))

        # Frontend — dense clustering
        self.cluster_eps = float(p("cluster_eps", 0.08))
        self.min_cluster_size = int(p("min_cluster_size", 2))

        # Frontend — shape filter
        self.enable_shape_filter = bool(p("enable_shape_filter", True))
        self.max_cluster_x_span = float(p("max_cluster_x_span", 0.32))
        self.max_cluster_y_span = float(p("max_cluster_y_span", 0.32))
        self.max_cluster_xy_span = float(p("max_cluster_xy_span", 0.26))
        self.min_cluster_z_span = float(p("min_cluster_z_span", 0.12))
        self.max_cluster_z_span = float(p("max_cluster_z_span", 0.90))
        self.max_cluster_xy_std = float(p("max_cluster_xy_std", 0.06))
        self.min_cluster_aspect_ratio = float(p("min_cluster_aspect_ratio", 1.0))
        self.max_cluster_aspect_ratio = float(p("max_cluster_aspect_ratio", 12.0))
        self.min_cluster_mean_intensity = float(p("min_cluster_mean_intensity", 110.0))
        self.min_z_bands = int(p("min_z_bands", 2))
        self.z_band_step = float(p("z_band_step", 0.05))

        # Frontend — scoring
        self.state_stable_confidence = float(p("state_stable_confidence", 0.74))
        self.state_degraded_confidence = float(p("state_degraded_confidence", 0.62))
        self.state_min_shape_score = float(p("state_min_shape_score", 0.45))
        self.state_min_intensity_score = float(p("state_min_intensity_score", 0.35))
        self.state_min_count_score = float(p("state_min_count_score", 0.10))

        # Candidate fusion
        self.candidate_fusion_xy_threshold = float(p("candidate_fusion_xy_threshold", 0.12))
        self.cluster_merge_xy = float(p("cluster_merge_xy", 0.15))

        # Sparse detector
        sparse_params = SparseParams(
            enable=bool(p("enable_sparse_column_mode", False)),
            xy_radius=float(p("sparse_column_xy_radius", 0.09)),
            min_range=float(p("sparse_column_min_range", 2.0)),
            min_points=int(p("sparse_column_min_points", 2)),
            min_z_range=float(p("sparse_column_min_z_range", 0.12)),
            max_xy_range=float(p("sparse_column_max_xy_range", 0.18)),
            max_xy_std=float(p("sparse_column_max_xy_std", 0.055)),
            min_z_bands=int(p("sparse_column_min_z_bands", 2)),
            min_mean_intensity=float(p("sparse_column_min_mean_intensity", 108.0)),
            near_sparse_enable=bool(p("near_sparse_enable", True)),
            near_sparse_min_range=float(p("near_sparse_min_range", 1.2)),
            near_sparse_max_range=float(p("near_sparse_max_range", 2.5)),
            near_sparse_min_points=int(p("near_sparse_min_points", 2)),
            near_sparse_min_z_range=float(p("near_sparse_min_z_range", 0.14)),
            near_sparse_max_xy_range=float(p("near_sparse_max_xy_range", 0.16)),
            near_sparse_max_xy_std=float(p("near_sparse_max_xy_std", 0.035)),
            near_sparse_min_z_bands=int(p("near_sparse_min_z_bands", 3)),
            near_sparse_min_mean_intensity=float(p("near_sparse_min_mean_intensity", 108.0)),
            near_sparse_min_contrast_ratio=float(p("near_sparse_min_contrast_ratio", 1.10)),
            z_band_step=self.z_band_step,
            min_aspect_ratio=self.min_cluster_aspect_ratio,
        )
        self.sparse_detector = SparseDetector(sparse_params)

        # IMU deskew
        self.enable_imu_deskew = bool(p("enable_imu_deskew", False))
        self.max_imu_deskew_shift = float(p("max_imu_deskew_shift", 0.5))
        self._deskewer = ImuDeskew(
            scan_duration=float(p("imu_deskew_scan_duration", 0.1)),
            time_scale=float(p("imu_deskew_time_scale", 1.0)),
        )
        self.imu_deskew_time_scale = float(p("imu_deskew_time_scale", 1.0))

        # Tracker
        tracker_params = TrackerParams(
            match_distance_threshold=float(p("match_distance_threshold", 0.35)),
            max_missing_frames=int(p("max_missing_frames", 6)),
            dense_stable_grace_frames=int(p("dense_stable_grace_frames", 3)),
            dense_degrade_window=int(p("dense_degrade_window", 5)),
            dense_degrade_max_hits=int(p("dense_degrade_max_hits", 1)),
            dense_degrade_min_confidence=float(p("dense_degrade_min_confidence", 0.56)),
            dense_recover_window=int(p("dense_recover_window", 3)),
            dense_recover_min_hits=int(p("dense_recover_min_hits", 2)),
            dense_recover_min_confidence=float(p("dense_recover_min_confidence", 0.64)),
            sparse_stable_window=int(p("sparse_stable_window", 5)),
            sparse_stable_min_hits=int(p("sparse_stable_min_hits", 2)),
            sparse_stable_min_confidence=float(p("sparse_stable_min_confidence", 0.54)),
            sparse_degrade_window=int(p("sparse_degrade_window", 7)),
            sparse_degrade_max_hits=int(p("sparse_degrade_max_hits", 1)),
            sparse_degrade_min_confidence=float(p("sparse_degrade_min_confidence", 0.44)),
            sparse_recover_window=int(p("sparse_recover_window", 4)),
            sparse_recover_min_hits=int(p("sparse_recover_min_hits", 2)),
            sparse_recover_min_confidence=float(p("sparse_recover_min_confidence", 0.60)),
            sparse_stable_grace_frames=int(p("sparse_stable_grace_frames", 4)),
            enable_output_hold=bool(p("enable_output_hold", True)),
            output_hold_frames=int(p("output_hold_frames", 3)),
            output_hold_max_missing=int(p("output_hold_max_missing", 3)),
            adaptive_rotation_tracking=bool(p("adaptive_rotation_tracking", True)),
            rotation_rate_threshold=float(p("rotation_rate_threshold", 0.35)),
            rotation_gate_expand=float(p("rotation_gate_expand", 1.35)),
            rotation_missing_bonus=int(p("rotation_missing_bonus", 2)),
            rotation_rebind_relax=float(p("rotation_rebind_relax", 0.25)),
            enable_imu_weak_prediction=bool(p("enable_imu_weak_prediction", True)),
            weak_prediction_velocity_gain=float(p("weak_prediction_velocity_gain", 0.6)),
            weak_prediction_gate_expand=float(p("weak_prediction_gate_expand", 1.35)),
            global_assignment_max_candidates=int(p("global_assignment_max_candidates", 14)),
        )
        self.tracker = Tracker(tracker_params)

        rospy.loginfo("V3 params: intensity_thr=%.0f cluster_eps=%.3f min_cluster=%d "
                      "sparse=%s near_sparse=%s",
                      self.intensity_threshold, self.cluster_eps, self.min_cluster_size,
                      sparse_params.enable, sparse_params.near_sparse_enable)

    def _setup_pubsub(self):
        # Input subscribers
        self._lock = threading.Lock()
        self._pending_cloud = None
        self._pending_event = threading.Event()
        self._pending_token = 0
        self._received_count = 0
        self._last_processed_token = -1
        self._dropped_frame_count = 0

        self._sub_cloud = rospy.Subscriber(
            self.input_topic, PointCloud2, self._cloud_cb, queue_size=10)
        self._sub_odom = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        self._sub_imu = rospy.Subscriber(
            self.imu_topic, Imu, self._imu_cb, queue_size=100)

        rospy.loginfo("V3 topics: lidar=%s odom=%s imu=%s",
                      self.input_topic, self.odom_topic, self.imu_topic)

        # Output publishers
        frame = self.frame_id_override or "rslidar"
        self._pub_obs = rospy.Publisher(
            "/reflector_observations", PoseArray, queue_size=10)
        self._pub_detail = rospy.Publisher(
            "/reflector_observations_detailed", ReflectorObservationArray, queue_size=10)
        self._pub_filtered = rospy.Publisher(
            "/filtered_pointcloud", PointCloud2, queue_size=10)
        self._pub_processed_input = rospy.Publisher(
            "/reflector_detector/processed_input_cloud", PointCloud2, queue_size=1)
        self._pub_input_cloud = rospy.Publisher(
            "/reflector_detector/input_cloud", PointCloud2, queue_size=1)
        self._pub_markers = rospy.Publisher(
            "/reflector_markers", MarkerArray, queue_size=10)

    def _init_state(self):
        self._frame_count = 0
        # Odometry state
        self._odom_curr = None       # latest odom msg
        self._odom_prev = None       # previous odom msg
        self._odom_prev_cloud_stamp = None
        self._odom_curr_cloud_stamp = None
        # IMU state
        self._imu_msgs = []          # buffered IMU messages between frames

    # ── Callbacks ──────────────────────────────────────────────────
    def _cloud_cb(self, msg):
        with self._lock:
            self._received_count += 1
            self._pending_cloud = msg
            self._pending_token = self._received_count
            self._pending_event.set()

    def _odom_cb(self, msg):
        with self._lock:
            self._odom_prev = self._odom_curr
            self._odom_curr = msg

    def _imu_cb(self, msg):
        self._deskewer.add_imu(msg)
        with self._lock:
            self._imu_msgs.append(msg)
            if len(self._imu_msgs) > 500:
                self._imu_msgs = self._imu_msgs[-200:]

    # ── Processing loop (worker thread) ────────────────────────────
    def _processing_loop(self):
        while not rospy.is_shutdown():
            self._pending_event.wait(0.1)
            if rospy.is_shutdown():
                return
            if not self._pending_event.is_set():
                continue
            with self._lock:
                msg = self._pending_cloud
                token = self._pending_token
                self._pending_cloud = None
                self._pending_event.clear()
                # Snapshot odom/IMU
                odom_curr = self._odom_curr
                odom_prev = self._odom_prev
                imu_buf = list(self._imu_msgs)
                self._imu_msgs = []

            if msg is None:
                continue
            skipped = max(0, token - self._last_processed_token - 1)
            self._dropped_frame_count += skipped
            self._last_processed_token = token
            self._process(msg, odom_prev, odom_curr, imu_buf)

    def _process(self, msg, odom_prev, odom_curr, imu_buf):
        t_start = time.monotonic()
        self._frame_count += 1

        # ── Step 1: Extract filtered points (V2 numpy pipeline) ──
        if self.publish_processed_input_cloud:
            self._publish_processed_input(msg)
        filtered = self._extract_filtered_points(msg)
        t_filter = time.monotonic()

        # ── Step 2: Dense clustering ──
        dense_candidates, used_indices = self._generate_dense_candidates(filtered)
        t_dense = time.monotonic()

        # ── Step 3: Sparse column detection ──
        sparse_candidates = self.sparse_detector.detect(filtered, used_indices)
        t_sparse = time.monotonic()

        # ── Step 4: Merge nearby dense candidates ──
        self._merge_nearby(dense_candidates)
        self._merge_nearby(sparse_candidates)

        # ── Step 5: Candidate fusion ──
        fused = self._fuse_candidates(dense_candidates, sparse_candidates)
        t_fuse = time.monotonic()

        # ── Step 6: Build odom/IMU state ──
        odom_state = self._build_odom_state(odom_prev, odom_curr, msg)
        imu_state = self._build_imu_state(imu_buf, msg)
        stamp_sec = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else time.time()

        # ── Step 7: Temporal tracking ──
        outputs = self.tracker.update(fused, odom_state, imu_state, stamp_sec)
        t_track = time.monotonic()

        # ── Step 8: Publish ──
        header = Header(stamp=msg.header.stamp,
                        frame_id=self.frame_id_override or msg.header.frame_id)
        self._publish(header, outputs)
        if self.publish_filtered_cloud:
            self._publish_filtered(header, filtered)

        # Debug: always print first 30 frames for diagnosis
        if self._frame_count <= 30:
            published_n = sum(1 for o in outputs if o.publish_eligible)
            rospy.loginfo(
                "V3 diag frame=%d pts=%d dense=%d sparse=%d fused=%d tracks=%d published=%d",
                self._frame_count, len(filtered), len(dense_candidates),
                len(sparse_candidates), len(fused), len(outputs), published_n)

        # Timing log
        if self.enable_timing_log and self._frame_count % self.debug_print_every_n == 0:
            filter_ms = (t_filter - t_start) * 1000
            dense_ms = (t_dense - t_filter) * 1000
            sparse_ms = (t_sparse - t_dense) * 1000
            fuse_ms = (t_fuse - t_sparse) * 1000
            track_ms = (t_track - t_fuse) * 1000
            total_ms = (t_track - t_start) * 1000
            stable_n = sum(1 for o in outputs if o.state == ObservationState.STABLE)
            published_n = sum(1 for o in outputs if o.publish_eligible)
            rospy.loginfo(
                "V3 timing frame=%d pts=%d candidates=%d stable=%d published=%d "
                "filter=%.1fms dense=%.1fms sparse=%.1fms fuse=%.1fms track=%.1fms total=%.1fms",
                self._frame_count, len(filtered), len(fused), stable_n, published_n,
                filter_ms, dense_ms, sparse_ms, fuse_ms, track_ms, total_ms)

    # ── Frontend: Point Extraction ─────────────────────────────────
    def _extract_filtered_points(self, msg):
        pts, offsets = self._extract_numpy(msg)
        if pts is not None:
            # Apply IMU deskew to filtered points (not full cloud)
            if self.enable_imu_deskew and len(pts) > 0 and offsets is not None:
                stamp_sec = msg.header.stamp.to_sec()
                if stamp_sec > 0:
                    deskewed = self._deskewer.deskew(pts, offsets, stamp_sec)
                    shift = np.linalg.norm(deskewed[:, :3] - pts[:, :3], axis=1)
                    max_shift = float(np.max(shift)) if len(shift) else 0.0
                    if np.isfinite(max_shift) and max_shift <= self.max_imu_deskew_shift:
                        pts = deskewed
                    else:
                        rospy.logwarn_throttle(
                            2.0,
                            "V3 IMU deskew rejected: max point shift %.3fm exceeds %.3fm; using raw points",
                            max_shift, self.max_imu_deskew_shift)
            return pts
        pts_iter, _ = self._extract_iter(msg)
        return pts_iter

    @staticmethod
    def _field_numpy_dtype(field, is_bigendian):
        endian = ">" if is_bigendian else "<"
        if field.datatype == PointField.FLOAT32:
            return endian + "f4"
        if field.datatype == PointField.FLOAT64:
            return endian + "f8"
        if field.datatype == PointField.UINT8:
            return "u1"
        if field.datatype == PointField.INT8:
            return "i1"
        if field.datatype == PointField.UINT16:
            return endian + "u2"
        if field.datatype == PointField.INT16:
            return endian + "i2"
        if field.datatype == PointField.UINT32:
            return endian + "u4"
        if field.datatype == PointField.INT32:
            return endian + "i4"
        return None

    def _extract_numpy(self, msg):
        count = msg.width * msg.height
        if count <= 0 or not msg.data:
            return np.empty((0, 4), dtype=np.float32), None
        fields = {f.name: f for f in msg.fields}
        field_offsets = {f.name: f.offset for f in msg.fields}
        for req in ("x", "y", "z", "intensity"):
            if req not in field_offsets:
                return None, None
        dtype_x = self._field_numpy_dtype(fields["x"], msg.is_bigendian)
        dtype_y = self._field_numpy_dtype(fields["y"], msg.is_bigendian)
        dtype_z = self._field_numpy_dtype(fields["z"], msg.is_bigendian)
        dtype_i = self._field_numpy_dtype(fields["intensity"], msg.is_bigendian)
        if dtype_x is None or dtype_y is None or dtype_z is None or dtype_i is None:
            return None, None
        try:
            x = np.ndarray((count,), dtype=dtype_x, buffer=msg.data,
                           offset=field_offsets["x"], strides=(msg.point_step,))
            y = np.ndarray((count,), dtype=dtype_y, buffer=msg.data,
                           offset=field_offsets["y"], strides=(msg.point_step,))
            z = np.ndarray((count,), dtype=dtype_z, buffer=msg.data,
                           offset=field_offsets["z"], strides=(msg.point_step,))
            intensity = np.ndarray((count,), dtype=dtype_i, buffer=msg.data,
                                   offset=field_offsets["intensity"], strides=(msg.point_step,))
        except (BufferError, TypeError, ValueError):
            return None, None

        # Extract per-point timestamps if available (Mid-360 has 'time' field)
        point_offsets = None
        if self.enable_imu_deskew:
            for f in msg.fields:
                if f.name in ("time", "t", "timestamp"):
                    dtype_t = self._field_numpy_dtype(f, msg.is_bigendian)
                    if dtype_t is None:
                        continue
                    t_arr = np.ndarray((count,), dtype=dtype_t, buffer=msg.data,
                                       offset=f.offset, strides=(msg.point_step,))
                    point_times = t_arr.astype(np.float64, copy=True)
                    if f.name == "timestamp":
                        # RoboSense stores absolute timestamps in nanoseconds.
                        # Deskew expects seconds from scan start.
                        if np.nanmedian(np.abs(point_times)) > 1.0e12:
                            point_times *= 1.0e-9
                        point_offsets = point_times - msg.header.stamp.to_sec()
                    else:
                        point_offsets = point_times * self.imu_deskew_time_scale
                    point_offsets = np.clip(point_offsets, 0.0, 2.0 * self._deskewer.scan_duration)
                    break

        range_sq = x*x + y*y + z*z
        mask = (np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(intensity))
        if self.min_detection_range > 0:
            mask &= range_sq >= self.min_detection_range * self.min_detection_range
        mask &= intensity >= self.intensity_threshold
        mask &= (z >= self.z_min) & (z <= self.z_max)
        if self.max_detection_range > 0:
            mask &= range_sq <= self.max_detection_range * self.max_detection_range
        if not np.any(mask):
            return np.empty((0, 4), dtype=np.float32), None

        filtered = np.column_stack((x[mask], y[mask], z[mask], intensity[mask]))
        filtered_offsets = point_offsets[mask] if point_offsets is not None else None
        return filtered.astype(np.float32, copy=False), filtered_offsets

    def _extract_iter(self, msg):
        """Slow fallback point extraction (no deskew, no timestamps)."""
        import sensor_msgs.point_cloud2 as pc2
        pts = []
        for p in pc2.read_points(msg, field_names=("x","y","z","intensity"), skip_nans=True):
            r2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2]
            if self.min_detection_range > 0 and r2 < self.min_detection_range**2:
                continue
            if p[3] < self.intensity_threshold:
                continue
            if p[2] < self.z_min or p[2] > self.z_max:
                continue
            if self.max_detection_range > 0 and r2 > self.max_detection_range**2:
                continue
            pts.append(p)
        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 4), dtype=np.float32), None

    # ── Frontend: Dense candidate generation ───────────────────────
    def _generate_dense_candidates(self, filtered):
        if len(filtered) == 0:
            return [], set()

        pts = filtered
        if self.cluster_voxel_size > 0.0:
            pts = self._voxel_downsample(pts, self.cluster_voxel_size)

        clusters = _cluster_xy(pts, self.cluster_eps, self.min_cluster_size)
        candidates = []
        used_indices = set()

        for indices in clusters:
            if len(indices) < self.min_cluster_size:
                continue
            cluster = pts[np.asarray(indices, dtype=np.intp)]
            xyz = cluster[:, :3]

            metrics = self._compute_shape_metrics(xyz)
            metrics["mean_intensity"] = float(np.mean(cluster[:, 3]))

            if self.enable_shape_filter and not self._is_shape_ok(metrics):
                continue

            center = np.median(xyz, axis=0)
            center_z = 0.5 * (metrics["z_min"] + metrics["z_max"])
            r = float(np.sqrt(center[0]**2 + center[1]**2 + center_z**2))
            score_i = self._score_intensity(metrics["mean_intensity"])
            score_s = self._score_shape(metrics)
            score_c = self._score_count(len(indices))
            score_r = self._score_range(r)

            cb = ClusterBounds(
                center_x=float(center[0]), center_y=float(center[1]), center_z=float(center_z),
                x_min=float(metrics["x_min"]), x_max=float(metrics["x_max"]),
                y_min=float(metrics["y_min"]), y_max=float(metrics["y_max"]),
                z_min=float(metrics["z_min"]), z_max=float(metrics["z_max"]),
                x_range=float(metrics["x_span"]), y_range=float(metrics["y_span"]),
                z_range=float(metrics["z_span"]), xy_range=float(metrics["xy_span"]),
                mean_intensity=metrics["mean_intensity"],
                std_xy=float(metrics["xy_std"]),
                z_band_count=metrics["z_band_count"],
                point_count=len(indices),
                aspect_ratio=float(metrics["aspect_ratio"]),
                score_intensity=score_i, score_shape=score_s,
                source_mask=SOURCE_DENSE, accepted_dense=True,
            )
            candidates.append(cb)
            for idx in indices:
                used_indices.add(idx)

        return candidates, used_indices

    def _compute_shape_metrics(self, xyz):
        mins = np.min(xyz, axis=0)
        maxs = np.max(xyz, axis=0)
        x_span = float(maxs[0] - mins[0])
        y_span = float(maxs[1] - mins[1])
        z_span = float(maxs[2] - mins[2])
        xy_span = max(x_span, y_span)
        var_xy = float(np.var(xyz[:, 0]) + np.var(xyz[:, 1]))
        xy_std = math.sqrt(max(0.0, var_xy))
        aspect = z_span / max(xy_span, 0.05)
        z_step = max(self.z_band_step, 0.01)
        z_bins = np.floor((xyz[:, 2] - mins[2]) / z_step).astype(np.int32)
        z_bands = int(np.unique(z_bins).size)
        return {
            "x_min": float(mins[0]), "x_max": float(maxs[0]),
            "y_min": float(mins[1]), "y_max": float(maxs[1]),
            "z_min": float(mins[2]), "z_max": float(maxs[2]),
            "x_span": x_span, "y_span": y_span, "z_span": z_span,
            "xy_span": xy_span, "xy_std": xy_std,
            "aspect_ratio": aspect, "z_band_count": z_bands,
        }

    def _is_shape_ok(self, m):
        if m["x_span"] > self.max_cluster_x_span: return False
        if m["y_span"] > self.max_cluster_y_span: return False
        if m["xy_span"] > self.max_cluster_xy_span: return False
        if m["z_span"] < self.min_cluster_z_span: return False
        if m["z_span"] > self.max_cluster_z_span: return False
        if m["xy_std"] > self.max_cluster_xy_std: return False
        if m["aspect_ratio"] < self.min_cluster_aspect_ratio: return False
        if m["aspect_ratio"] > self.max_cluster_aspect_ratio: return False
        if self.min_cluster_mean_intensity > 0 and m.get("mean_intensity", 0) < self.min_cluster_mean_intensity:
            return False
        if self.min_z_bands > 0 and m.get("z_band_count", 0) < self.min_z_bands:
            return False
        # Z-dominance
        if m["z_span"] < m["x_span"] or m["z_span"] < m["y_span"]:
            return False
        return True

    def _score_intensity(self, mean_i):
        base = self.min_cluster_mean_intensity if self.min_cluster_mean_intensity > 0 else self.intensity_threshold
        ramp = max(10.0, 0.1 * max(base, 100.0))
        return float(np.clip((mean_i - base) / ramp, 0.0, 1.0))

    def _score_shape(self, m):
        s_xy = max(0.0, 1.0 - m["xy_span"] / max(self.max_cluster_xy_span, 1e-6))
        s_std = max(0.0, 1.0 - m["xy_std"] / max(self.max_cluster_xy_std, 1e-6))
        z_den = max(self.max_cluster_z_span - self.min_cluster_z_span, 1e-6)
        s_z = min(1.0, max(0.0, (m["z_span"] - self.min_cluster_z_span) / z_den))
        a_den = max(self.max_cluster_aspect_ratio - self.min_cluster_aspect_ratio, 1e-6)
        s_a = min(1.0, max(0.0, (m["aspect_ratio"] - self.min_cluster_aspect_ratio) / a_den))
        if self.min_z_bands > 0:
            s_band = min(1.0, m.get("z_band_count", 0) / float(max(self.min_z_bands, 3)))
        else:
            s_band = 1.0
        return 0.30*s_xy + 0.25*s_std + 0.20*s_z + 0.15*s_a + 0.10*s_band

    def _score_count(self, n):
        span = max(4.0, float(self.min_cluster_size * 3))
        return float(np.clip((n - self.min_cluster_size + 1) / span, 0.0, 1.0))

    def _score_range(self, r):
        if self.max_detection_range <= 0:
            return 1.0
        soft = min(self.max_detection_range, max(8.0, self.max_detection_range * 0.4))
        if r <= soft:
            return 1.0
        tail = max(self.max_detection_range - soft, 1.0)
        return float(np.clip(1.0 - 0.8 * (r - soft) / tail, 0.2, 1.0))

    def _voxel_downsample(self, pts, vsize):
        if len(pts) < 2 or vsize <= 0:
            return pts
        vox_indices = np.floor(pts[:, :3] / vsize).astype(np.int32)
        order = np.argsort(pts[:, 3])[::-1]  # descending intensity
        seen = set()
        keep = []
        for i in order:
            key = (vox_indices[i, 0], vox_indices[i, 1], vox_indices[i, 2])
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return pts[np.array(sorted(keep), dtype=np.intp)]

    # ── Merge & Fusion ────────────────────────────────────────────
    def _merge_nearby(self, candidates):
        """Merge candidates within cluster_merge_xy."""
        if len(candidates) < 2:
            return
        merged = True
        while merged:
            merged = False
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    a, b = candidates[i], candidates[j]
                    dx = a.center_x - b.center_x
                    dy = a.center_y - b.center_y
                    if dx*dx + dy*dy <= self.cluster_merge_xy**2:
                        # Keep the one with more points
                        if b.point_count > a.point_count:
                            candidates[i], candidates[j] = candidates[j], candidates[i]
                        # Merge b into a
                        self._merge_bounds(a, b)
                        candidates.pop(j)
                        merged = True
                        break
                if merged:
                    break

    def _fuse_candidates(self, dense, sparse):
        """Fuse sparse into dense candidates, ported from V1."""
        fused = list(dense)
        for sc in sparse:
            merged = False
            for fc in fused:
                dx = sc.center_x - fc.center_x
                dy = sc.center_y - fc.center_y
                if dx*dx + dy*dy <= self.candidate_fusion_xy_threshold**2:
                    if fc.source_mask & SOURCE_DENSE:
                        self._merge_bounds(fc, sc)
                        fc.source_mask |= SOURCE_SPARSE
                        fc.accepted_sparse = True
                    else:
                        self._merge_bounds(fc, sc)
                    merged = True
                    break
            if not merged:
                fused.append(sc)
        return fused

    @staticmethod
    def _copy_bounds(dst, src):
        dst.center_x = src.center_x
        dst.center_y = src.center_y
        dst.center_z = src.center_z
        dst.x_min = src.x_min
        dst.x_max = src.x_max
        dst.y_min = src.y_min
        dst.y_max = src.y_max
        dst.z_min = src.z_min
        dst.z_max = src.z_max
        dst.x_range = src.x_range
        dst.y_range = src.y_range
        dst.z_range = src.z_range
        dst.xy_range = src.xy_range
        dst.mean_intensity = src.mean_intensity
        dst.std_xy = src.std_xy
        dst.z_band_count = src.z_band_count
        dst.point_count = src.point_count
        dst.aspect_ratio = src.aspect_ratio
        dst.score_intensity = src.score_intensity
        dst.score_shape = src.score_shape
        dst.source_mask = src.source_mask
        dst.accepted_dense = src.accepted_dense
        dst.accepted_sparse = src.accepted_sparse
        dst.near_sparse_rescue = src.near_sparse_rescue

    @staticmethod
    def _merge_bounds(a, b):
        total = max(1, a.point_count + b.point_count)
        a.center_x = (a.center_x * a.point_count + b.center_x * b.point_count) / total
        a.center_y = (a.center_y * a.point_count + b.center_y * b.point_count) / total
        a.x_min = min(a.x_min, b.x_min)
        a.x_max = max(a.x_max, b.x_max)
        a.y_min = min(a.y_min, b.y_min)
        a.y_max = max(a.y_max, b.y_max)
        a.z_min = min(a.z_min, b.z_min)
        a.z_max = max(a.z_max, b.z_max)
        a.x_range = a.x_max - a.x_min
        a.y_range = a.y_max - a.y_min
        a.z_range = a.z_max - a.z_min
        a.xy_range = max(a.x_range, a.y_range)
        a.center_z = 0.5 * (a.z_min + a.z_max)
        a.mean_intensity = (a.mean_intensity * a.point_count + b.mean_intensity * b.point_count) / total
        a.std_xy = max(a.std_xy, b.std_xy)
        a.z_band_count = max(a.z_band_count, b.z_band_count)
        a.point_count = total
        a.aspect_ratio = a.z_range / max(a.xy_range, 0.01)
        a.score_intensity = max(a.score_intensity, b.score_intensity)
        a.score_shape = max(a.score_shape, b.score_shape)
        a.source_mask |= b.source_mask
        a.accepted_dense = a.accepted_dense or b.accepted_dense
        a.accepted_sparse = a.accepted_sparse or b.accepted_sparse
        a.near_sparse_rescue = a.near_sparse_rescue or b.near_sparse_rescue

    # ── Odom/IMU State ─────────────────────────────────────────────
    def _build_odom_state(self, prev, curr, cloud_msg):
        state = OdometryState()
        if curr is None:
            return state
        state.valid = True
        p = curr.pose.pose.position
        q = curr.pose.pose.orientation
        state.T_curr = self._pose_to_matrix(p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        cov = curr.pose.covariance
        state.covariance_xy_trace = cov[0] + cov[7] if len(cov) >= 36 else 0.0

        if prev is not None:
            state.prev_valid = True
            pp = prev.pose.pose.position
            pq = prev.pose.pose.orientation
            state.T_prev = self._pose_to_matrix(pp.x, pp.y, pp.z, pq.x, pq.y, pq.z, pq.w)
            state.delta_xyz = np.array([
                p.x - pp.x, p.y - pp.y, p.z - pp.z
            ])
        return state

    @staticmethod
    def _quat_to_rot(qx, qy, qz, qw):
        n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n < 1e-12:
            return np.eye(3)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
        return np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
        ])

    def _pose_to_matrix(self, x, y, z, qx, qy, qz, qw):
        R = self._quat_to_rot(qx, qy, qz, qw)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def _build_imu_state(self, imu_buf, cloud_msg):
        state = ImuState()
        if not imu_buf:
            return state
        # Integrate angular velocity over buffer
        dt_total = 0.0
        accum_angle = 0.0
        for i in range(1, len(imu_buf)):
            dt = (imu_buf[i].header.stamp - imu_buf[i-1].header.stamp).to_sec()
            if dt <= 0 or dt > 0.5:
                continue
            dt_total += dt
            av = imu_buf[i].angular_velocity
            accum_angle += math.sqrt(av.x**2 + av.y**2 + av.z**2) * dt

        if dt_total > 0:
            # Use last IMU message's orientation for delta rotation (simplified)
            # Full IMU integration would need gyro integration
            state.valid = True
            state.dt = dt_total
            state.angular_velocity = accum_angle / dt_total if dt_total > 0 else 0.0

            # Simplified delta rotation from angular velocity
            last_imu = imu_buf[-1]
            av = last_imu.angular_velocity
            angle = math.sqrt(av.x**2 + av.y**2 + av.z**2) * state.dt
            if angle > 1e-6:
                axis = np.array([av.x, av.y, av.z]) / max(angle / state.dt, 1e-6)
                axis = axis / np.linalg.norm(axis)
                c = math.cos(angle)
                s = math.sin(angle)
                k = 1.0 - c
                x, y, z = axis
                state.delta_rotation = np.array([
                    [c + x*x*k, x*y*k - z*s, x*z*k + y*s],
                    [y*x*k + z*s, c + y*y*k, y*z*k - x*s],
                    [z*x*k - y*s, z*y*k + x*s, c + z*z*k],
                ])
        return state

    # ── Publishing ─────────────────────────────────────────────────
    def _publish(self, header, outputs):
        pa = PoseArray()
        pa.header = header
        detail = ReflectorObservationArray()
        detail.header = header

        for out in outputs:
            if not out.publish_eligible:
                continue
            pose = Pose()
            pose.position.x = out.center_x
            pose.position.y = out.center_y
            pose.position.z = out.center_z
            pose.orientation.w = 1.0
            pa.poses.append(pose)

            obs = ReflectorObservation()
            obs.tracker_id = out.tracker_id
            obs.pose = pose
            obs.confidence = float(out.confidence)
            obs.state = int(out.state)
            obs.gate_mode = 0
            obs.stable_count = min(out.stable_count, 255)
            obs.missing_count = out.missing_count
            obs.gate_radius = out.gate_radius
            obs.score_intensity = out.score_intensity
            obs.score_shape = out.score_shape
            obs.score_temporal = out.score_temporal
            obs.score_prediction = out.score_prediction
            obs.association_cost = out.association_cost
            detail.observations.append(obs)

        self._pub_obs.publish(pa)
        self._pub_detail.publish(detail)

        if self.enable_visualization:
            self._publish_markers(header, outputs)

    def _publish_markers(self, header, outputs):
        ma = MarkerArray()
        # Delete all previous markers
        del_m = Marker()
        del_m.header = header
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)

        for out in outputs:
            if not out.publish_eligible:
                continue
            state = out.state
            # Only stable/degraded outputs are published; tentative tracks stay internal.
            if state == ObservationState.STABLE:
                r, g, b, a = 0.0, 1.0, 0.0, 0.8
            elif state == ObservationState.DEGRADED:
                r, g, b, a = 1.0, 1.0, 0.0, 0.6
            else:
                r, g, b, a = 0.5, 0.5, 0.5, 0.4

            m = Marker()
            m.header = header
            m.ns = "reflector_boxes"
            m.id = out.tracker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = out.center_x
            m.pose.position.y = out.center_y
            m.pose.position.z = out.center_z
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = a
            m.lifetime = rospy.Duration(0.5)
            ma.markers.append(m)

        self._pub_markers.publish(ma)

    def _publish_filtered(self, header, filtered):
        if len(filtered) == 0:
            msg = PointCloud2()
            msg.header = header
            self._pub_filtered.publish(msg)
            return
        # Pack as PointCloud2 with x/y/z/rgb
        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('intensity', np.float32),
        ])
        arr = np.zeros(len(filtered), dtype=dtype)
        arr['x'] = filtered[:, 0]
        arr['y'] = filtered[:, 1]
        arr['z'] = filtered[:, 2]
        arr['intensity'] = filtered[:, 3]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(arr)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_bigendian = False
        msg.is_dense = True
        msg.data = arr.tobytes()
        self._pub_filtered.publish(msg)

    def _publish_processed_input(self, msg):
        snapshot = PointCloud2()
        snapshot.header = Header(stamp=msg.header.stamp,
                                 frame_id=self.frame_id_override or msg.header.frame_id)
        snapshot.height = msg.height
        snapshot.width = msg.width
        snapshot.fields = msg.fields
        snapshot.is_bigendian = msg.is_bigendian
        snapshot.point_step = msg.point_step
        snapshot.row_step = msg.row_step
        snapshot.is_dense = msg.is_dense
        snapshot.data = msg.data
        self._pub_processed_input.publish(snapshot)
        self._pub_input_cloud.publish(snapshot)


def main():
    rospy.init_node("reflector_detector_v3")
    ReflectorDetectorV3()
    rospy.spin()


if __name__ == "__main__":
    main()
