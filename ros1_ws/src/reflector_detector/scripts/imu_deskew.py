#!/usr/bin/env python3
"""IMU motion compensation (deskew) for LiDAR point clouds.

Ported from V1 C++ imu_deskew.hpp, which follows the FASTER-LIO approach:
  1. Buffer IMU (gyro + accel) messages
  2. Forward-integrate to get IMU-frame poses
  3. For each LiDAR point: interpolate pose at point timestamp,
     transform point from its scan time to scan-end time.

Performance note: deskew is applied to FILTERED points (~500/frame),
not the full cloud (~20k/frame), keeping cost negligible.
"""
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np


class ImuDeskew:
    """FASTER-LIO-style IMU deskew for Mid-360 and similar LiDARs."""

    G = 9.81

    def __init__(self, scan_duration: float = 0.1, time_scale: float = 1.0):
        self.scan_duration = scan_duration
        self.time_scale = time_scale
        self._imu_buf: deque = deque(maxlen=500)
        self._initialized = False
        self._mean_acc = np.zeros(3)
        self._mean_gyr = np.zeros(3)
        self._last_lidar_end_time = 0.0

    def add_imu(self, msg) -> None:
        """Call from IMU subscriber callback."""
        self._imu_buf.append(msg)

    def deskew(self,
               points: np.ndarray,          # (N, 4): [x, y, z, intensity]
               point_offsets: np.ndarray,    # (N,): per-point time offset from scan-start (sec)
               cloud_stamp_sec: float,       # scan-start ROS time (seconds)
               ) -> np.ndarray:
        """Return deskewed copy of points. If insufficient IMU, return input unchanged."""
        n = len(points)
        if n == 0:
            return points

        pcl_beg = cloud_stamp_sec
        if len(point_offsets) > 0:
            max_offset = float(np.max(point_offsets))
            pcl_end = pcl_beg + max(max_offset, 0.001)
        else:
            pcl_end = pcl_beg + self.scan_duration

        # Collect relevant IMU messages
        imu_start = self._last_lidar_end_time if self._last_lidar_end_time > 0 else pcl_beg - 0.05
        v_imu = [m for m in self._imu_buf
                 if imu_start <= m.header.stamp.to_sec() <= pcl_end + 0.02]
        if len(v_imu) < 2:
            self._last_lidar_end_time = pcl_end
            return points

        # First frame: estimate gravity direction
        if not self._initialized:
            acc_sum = np.zeros(3)
            gyr_sum = np.zeros(3)
            for m in v_imu:
                a = m.linear_acceleration
                g = m.angular_velocity
                acc_sum += [a.x, a.y, a.z]
                gyr_sum += [g.x, g.y, g.z]
            self._mean_acc = acc_sum / len(v_imu)
            self._mean_gyr = gyr_sum / len(v_imu)
            if np.linalg.norm(self._mean_acc) < 1e-6:
                self._last_lidar_end_time = pcl_end
                return points
            self._initialized = True

        # Forward-integrate IMU → poses
        poses = self._integrate_imu(v_imu, pcl_beg)
        if len(poses) < 2:
            self._last_lidar_end_time = pcl_end
            return points

        R_end = poses[-1][3]
        pos_end = poses[-1][1]

        # Compensate each point (descending time order)
        result = points.copy()
        order = np.argsort(-point_offsets) if len(point_offsets) > 0 else np.arange(n)[::-1]
        pose_idx = len(poses) - 1

        for idx in order:
            pt_t = point_offsets[idx] if len(point_offsets) > 0 else 0.0
            while pose_idx > 1 and pt_t <= poses[pose_idx - 1][0] + 1e-9:
                pose_idx -= 1

            head = poses[pose_idx - 1]
            tail = poses[pose_idx]
            dt_pt = pt_t - head[0]

            dR = _so3_exp(tail[5], dt_pt)
            R_i = head[3] @ dR
            T_ei = head[1] + head[2] * dt_pt + 0.5 * tail[4] * dt_pt * dt_pt - pos_end
            P_i = result[idx, :3]
            P_comp = R_end.T @ (R_i @ P_i + T_ei)
            result[idx, :3] = P_comp

        self._last_lidar_end_time = pcl_end
        return result

    def estimate_rotation(self, t_start: float, t_end: float) -> Tuple[bool, np.ndarray]:
        """Estimate relative rotation between two timestamps from IMU gyro.
        Returns (success, 3x3 rotation matrix)."""
        delta_rot = np.eye(3)
        if t_end <= t_start:
            return False, delta_rot

        v_imu = [m for m in self._imu_buf
                 if t_start - 0.01 <= m.header.stamp.to_sec() <= t_end + 0.01]
        if len(v_imu) < 2:
            return False, delta_rot

        for k in range(len(v_imu) - 1):
            t0 = v_imu[k].header.stamp.to_sec()
            t1 = v_imu[k + 1].header.stamp.to_sec()
            if t1 <= t_start or t0 >= t_end:
                continue
            dt = min(t1, t_end) - max(t0, t_start)
            if dt <= 0:
                continue
            gyro = np.array([
                0.5 * (v_imu[k].angular_velocity.x + v_imu[k + 1].angular_velocity.x),
                0.5 * (v_imu[k].angular_velocity.y + v_imu[k + 1].angular_velocity.y),
                0.5 * (v_imu[k].angular_velocity.z + v_imu[k + 1].angular_velocity.z),
            ])
            dR = _so3_exp(gyro, dt)
            delta_rot = delta_rot @ dR

        return True, delta_rot

    # ── internals ──────────────────────────────────────────────────
    def _integrate_imu(self, v_imu: list, pcl_beg: float) -> list:
        """Forward-integrate IMU measurements → list of (offset_t, pos, vel, rot, acc, gyr)."""
        poses = []
        pos = np.zeros(3)
        vel = np.zeros(3)
        rot = np.eye(3)

        poses.append((0.0, pos.copy(), vel.copy(), rot.copy(),
                      np.array([0.0, 0.0, -self.G]), self._mean_gyr.copy()))

        for k in range(len(v_imu) - 1):
            head = v_imu[k]
            tail = v_imu[k + 1]
            t_head = head.header.stamp.to_sec()
            t_tail = tail.header.stamp.to_sec()

            if t_head < self._last_lidar_end_time:
                continue

            a = head.linear_acceleration
            b = tail.linear_acceleration
            acc_avr = np.array([0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z)])

            gz = max(-self._mean_acc[2], 0.1)
            acc_avr = acc_avr * (self.G / gz)

            g = head.angular_velocity
            h = tail.angular_velocity
            gyr_avr = np.array([0.5 * (g.x + h.x), 0.5 * (g.y + h.y), 0.5 * (g.z + h.z)])

            dt = t_tail - t_head
            if dt <= 0:
                continue

            dR = _so3_exp(gyr_avr, dt)
            rot = rot @ dR
            acc_world = rot @ acc_avr - np.array([0.0, 0.0, self.G])
            vel = vel + acc_world * dt
            pos = pos + vel * dt + 0.5 * acc_world * dt * dt

            poses.append((t_tail - pcl_beg, pos.copy(), vel.copy(), rot.copy(),
                          acc_world.copy(), gyr_avr.copy()))

        return poses


def _so3_exp(omega: np.ndarray, dt: float) -> np.ndarray:
    """SO(3) exponential map: angular_velocity * dt → rotation matrix."""
    ang = float(np.linalg.norm(omega))
    if ang < 1e-7:
        return np.eye(3)
    r = omega / ang
    r_ang = ang * dt
    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    return np.eye(3) + math.sin(r_ang) * K + (1.0 - math.cos(r_ang)) * K @ K
