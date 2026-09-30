#!/usr/bin/env python3
"""Evaluate simple reflector-detection baselines directly from a raw rosbag.

The script samples the point cloud frame nearest to each manually labeled
ground-truth stamp, runs several deterministic high-intensity clustering
baselines, and reports the same XY matching metrics used for the paper.
"""

import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_gt(path):
    positives = defaultdict(list)
    negatives = []
    all_stamps = set()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamp = parse_float(row.get("stamp"))
            if stamp is None:
                continue
            all_stamps.add(stamp)
            visibility = (row.get("visibility") or "").strip().lower()
            if visibility in ("0", "false", "no"):
                negatives.append(stamp)
                continue
            x = parse_float(row.get("x"))
            y = parse_float(row.get("y"))
            z = parse_float(row.get("z"), 0.0)
            if x is None or y is None:
                continue
            positives[stamp].append(
                {
                    "stamp": stamp,
                    "gt_id": row.get("gt_id", ""),
                    "x": x,
                    "y": y,
                    "z": z,
                    "range_xy": math.hypot(x, y),
                }
            )
    return positives, sorted(set(negatives)), sorted(all_stamps)


def nearest_index(values, target):
    lo = np.searchsorted(values, target)
    candidates = []
    if lo < len(values):
        candidates.append(lo)
    if lo > 0:
        candidates.append(lo - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda idx: abs(values[idx] - target))


def pointcloud_to_array(msg):
    from sensor_msgs import point_cloud2

    rows = []
    for point in point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z", "intensity"),
        skip_nans=True,
    ):
        rows.append(point)
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def collect_labeled_clouds(bag_path, topic, stamps, tolerance):
    import rosbag

    target_stamps = np.asarray(stamps, dtype=float)
    best = {stamp: {"dt": float("inf"), "msg": None, "cloud_stamp": None} for stamp in stamps}

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            msg_stamp = float(msg.header.stamp.to_sec())
            idx = nearest_index(target_stamps, msg_stamp)
            if idx is None:
                continue
            target = float(target_stamps[idx])
            dt = abs(msg_stamp - target)
            if dt <= tolerance and dt < best[target]["dt"]:
                best[target] = {"dt": dt, "msg": msg, "cloud_stamp": msg_stamp}

    clouds = {}
    missing = []
    for stamp in stamps:
        item = best[stamp]
        if item["msg"] is None:
            missing.append(stamp)
            continue
        clouds[stamp] = {
            "cloud_stamp": item["cloud_stamp"],
            "dt": item["dt"],
            "points": pointcloud_to_array(item["msg"]),
        }
    return clouds, missing


def cluster_indices_xy(points_xy, eps):
    n = len(points_xy)
    if n == 0:
        return []
    tree = cKDTree(points_xy)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        cluster = []
        while stack:
            j = stack.pop()
            cluster.append(j)
            for nb in tree.query_ball_point(points_xy[j], eps):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(cluster)
    return clusters


def z_band_count(z_values, band_size=0.05):
    if len(z_values) == 0:
        return 0
    bands = np.floor((z_values - np.min(z_values)) / band_size).astype(int)
    return int(len(np.unique(bands)))


def cluster_features(cluster_points):
    xyz = cluster_points[:, :3]
    intensities = cluster_points[:, 3]
    center = np.mean(xyz, axis=0)
    xy = xyz[:, :2]
    xy_span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    xy_range = float(np.max(np.linalg.norm(xy - np.mean(xy, axis=0), axis=1))) if len(xy) else 0.0
    z_range = float(np.ptp(xyz[:, 2]))
    std_xy = float(np.mean(np.std(xy, axis=0))) if len(xy) else 0.0
    aspect = z_range / max(xy_span, 1e-6)
    return {
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
        "n_points": int(len(cluster_points)),
        "mean_intensity": float(np.mean(intensities)),
        "max_intensity": float(np.max(intensities)),
        "xy_span": xy_span,
        "xy_radius": xy_range,
        "z_range": z_range,
        "std_xy": std_xy,
        "aspect_ratio": float(aspect),
        "z_bands": z_band_count(xyz[:, 2]),
        "range_xy": float(math.hypot(center[0], center[1])),
    }


def high_points(points, threshold, z_min=-0.5, z_max=0.6, min_range=0.0, max_range=20.0):
    if len(points) == 0:
        return points
    xy_range = np.linalg.norm(points[:, :2], axis=1)
    mask = (
        np.isfinite(points).all(axis=1)
        & (points[:, 3] >= threshold)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
        & (xy_range >= min_range)
        & (xy_range <= max_range)
    )
    return points[mask]


def accept_fixed_threshold(feat):
    return feat["n_points"] >= 2


def accept_sorla_like(feat):
    return (
        feat["n_points"] >= 2
        and feat["xy_span"] <= 0.45
        and feat["std_xy"] <= 0.12
        and feat["range_xy"] <= 8.0
    )


def accept_dense_geometry(feat):
    min_points = 2 if feat["range_xy"] < 2.8 else 3
    return (
        feat["n_points"] >= min_points
        and feat["z_range"] >= 0.12
        and feat["xy_span"] <= 0.24
        and feat["std_xy"] <= 0.055
        and feat["aspect_ratio"] >= 1.0
        and feat["z_bands"] >= 2
    )


def accept_dense_only_strict(feat):
    return (
        feat["n_points"] >= 8
        and feat["z_range"] >= 0.16
        and feat["xy_span"] <= 0.24
        and feat["std_xy"] <= 0.055
        and feat["aspect_ratio"] >= 1.0
        and feat["z_bands"] >= 2
    )


def run_baseline(points, method, threshold, eps):
    strong = high_points(points, threshold)
    if len(strong) == 0:
        return []
    clusters = cluster_indices_xy(strong[:, :2], eps)
    rows = []
    for cluster in clusters:
        feat = cluster_features(strong[np.asarray(cluster, dtype=int)])
        if method == "fixed_threshold_cluster":
            keep = accept_fixed_threshold(feat)
        elif method == "sorla_like_reflective_cluster":
            keep = accept_sorla_like(feat)
        elif method == "single_frame_geometry":
            keep = accept_dense_geometry(feat)
        elif method == "dense_only_strict":
            keep = accept_dense_only_strict(feat)
        else:
            raise ValueError(f"Unknown method: {method}")
        if keep:
            rows.append(feat)
    return rows


def xy_distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def match_frame(gt_rows, det_rows, threshold):
    pairs = []
    for gi, gt in enumerate(gt_rows):
        for di, det in enumerate(det_rows):
            dist = xy_distance(gt, det)
            if dist <= threshold:
                pairs.append((dist, gi, di))
    pairs.sort()
    used_gt = set()
    used_det = set()
    matches = []
    for dist, gi, di in pairs:
        if gi in used_gt or di in used_det:
            continue
        used_gt.add(gi)
        used_det.add(di)
        matches.append((dist, gi, di))
    return len(matches), len(det_rows) - len(used_det), len(gt_rows) - len(used_gt), matches


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def range_bin(range_xy):
    if range_xy < 2.0:
        return "0.9-2m"
    if range_xy < 4.0:
        return "2-4m"
    return ">4m"


def evaluate_method(detections_by_stamp, positives, negatives, match_threshold):
    total_tp = total_fp = total_fn = 0
    bins = defaultdict(lambda: {"tp": 0, "fn": 0})
    match_details = []
    for stamp, gt_rows in sorted(positives.items()):
        det_rows = detections_by_stamp.get(stamp, [])
        tp, fp, fn, matches = match_frame(gt_rows, det_rows, match_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        matched_gt = {gi for _, gi, _ in matches}
        for gi, gt in enumerate(gt_rows):
            bucket = range_bin(gt["range_xy"])
            if gi in matched_gt:
                bins[bucket]["tp"] += 1
            else:
                bins[bucket]["fn"] += 1
        for dist, gi, di in matches:
            match_details.append(
                {
                    "stamp": stamp,
                    "gt_id": gt_rows[gi]["gt_id"],
                    "det_index": di,
                    "xy_error": dist,
                }
            )

    neg_frame_count = len(negatives)
    neg_fp_frames = 0
    neg_fp_total = 0
    for stamp in negatives:
        count = len(detections_by_stamp.get(stamp, []))
        neg_fp_total += count
        if count:
            neg_fp_frames += 1

    precision, recall, f1 = prf(total_tp, total_fp, total_fn)
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_frame_count": neg_frame_count,
        "negative_fp_frames": neg_fp_frames,
        "negative_fp_frame_rate": neg_fp_frames / neg_frame_count if neg_frame_count else 0.0,
        "negative_fp_total": neg_fp_total,
        "bins": bins,
        "match_details": match_details,
    }


def summarize_timing(values):
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
    }


def parse_proposed_timing(log_path):
    if not log_path or not os.path.exists(log_path):
        return {}
    pattern = re.compile(r"([A-Za-z_]+)=([0-9.]+)ms")
    rows = defaultdict(list)
    with open(log_path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "V3 timing" not in line:
                continue
            for key, value in pattern.findall(line):
                rows[key].append(float(value))
    return {key: summarize_timing(values) for key, values in rows.items()}


def read_proposed_metrics(metrics_csv, negative_json, log_path):
    if not metrics_csv or not os.path.exists(metrics_csv):
        return None
    with open(metrics_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
    row = {
        "method": "proposed_full",
        "description": "本文方法：强度+几何+dense/sparse+多帧状态跟踪",
        "threshold_intensity": "",
        "match_threshold_xy_m": float(first["threshold"]),
        "tp": int(first["tp"]),
        "fp": int(first["fp"]),
        "fn": int(first["fn"]),
        "precision": float(first["precision"]),
        "recall": float(first["recall"]),
        "f1": float(first["f1"]),
    }
    if negative_json and os.path.exists(negative_json):
        with open(negative_json, encoding="utf-8") as handle:
            neg = json.load(handle)
        row["negative_frame_count"] = neg.get("negative_frame_count", "")
        row["negative_fp_frames"] = neg.get("frames_with_false_positive", "")
        row["negative_fp_frame_rate"] = neg.get("false_positive_frame_rate", "")
        row["negative_fp_total"] = neg.get("total_false_positive_detections", "")
    timing = parse_proposed_timing(log_path)
    for key, stats in timing.items():
        if key == "total":
            row["mean_frame_ms"] = stats["mean_ms"]
            row["p95_frame_ms"] = stats["p95_ms"]
            row["max_frame_ms"] = stats["max_ms"]
    return row


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value):
    if isinstance(value, float):
        return "%.6f" % value
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lidar-topic", default="/rslidar_points")
    parser.add_argument("--frame-tolerance", type=float, default=0.08)
    parser.add_argument("--match-threshold-xy", type=float, default=0.15)
    parser.add_argument("--cluster-eps", type=float, default=0.08)
    parser.add_argument("--thresholds", default="100,120,140,160")
    parser.add_argument("--primary-threshold", type=float, default=140.0)
    parser.add_argument("--proposed-metrics", default="")
    parser.add_argument("--proposed-negative-json", default="")
    parser.add_argument("--proposed-log", default="")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    positives, negatives, all_stamps = load_gt(args.ground_truth)
    clouds, missing = collect_labeled_clouds(args.bag, args.lidar_topic, all_stamps, args.frame_tolerance)

    methods = [
        (
            "fixed_threshold_cluster",
            "固定强度阈值+XY聚类基线，无几何确认、无时序跟踪",
        ),
        (
            "sorla_like_reflective_cluster",
            "SORLA风格反光提取基线：高反点聚类+簇尺度过滤，无时序跟踪",
        ),
        (
            "single_frame_geometry",
            "去掉多帧跟踪的单帧几何候选：强度+局部柱状几何",
        ),
        (
            "dense_only_strict",
            "去掉sparse补检的dense-only消融：仅保留点数充足的柱状聚类",
        ),
    ]

    detection_rows = []
    summary_rows = []
    bin_rows = []
    timing_rows = []

    for method, description in methods:
        detections_by_stamp = {}
        timing_ms = []
        for stamp in all_stamps:
            item = clouds.get(stamp)
            if item is None:
                detections_by_stamp[stamp] = []
                continue
            start = time.perf_counter()
            detections = run_baseline(item["points"], method, args.primary_threshold, args.cluster_eps)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timing_ms.append(elapsed_ms)
            detections_by_stamp[stamp] = detections
            for idx, det in enumerate(detections):
                row = {
                    "method": method,
                    "stamp": "%.9f" % stamp,
                    "cloud_stamp": "%.9f" % item["cloud_stamp"],
                    "stamp_delta_s": "%.6f" % item["dt"],
                    "det_index": idx,
                }
                row.update({key: fmt(value) for key, value in det.items()})
                detection_rows.append(row)

        result = evaluate_method(detections_by_stamp, positives, negatives, args.match_threshold_xy)
        timing = summarize_timing(timing_ms)
        summary_rows.append(
            {
                "method": method,
                "description": description,
                "threshold_intensity": args.primary_threshold,
                "match_threshold_xy_m": args.match_threshold_xy,
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "precision": fmt(result["precision"]),
                "recall": fmt(result["recall"]),
                "f1": fmt(result["f1"]),
                "negative_frame_count": result["negative_frame_count"],
                "negative_fp_frames": result["negative_fp_frames"],
                "negative_fp_frame_rate": fmt(result["negative_fp_frame_rate"]),
                "negative_fp_total": result["negative_fp_total"],
                "mean_frame_ms": fmt(timing["mean_ms"]),
                "p95_frame_ms": fmt(timing["p95_ms"]),
                "max_frame_ms": fmt(timing["max_ms"]),
            }
        )
        timing_rows.append({"method": method, **{key: fmt(value) for key, value in timing.items()}})
        for bucket, counts in sorted(result["bins"].items()):
            recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
            bin_rows.append(
                {
                    "method": method,
                    "range_bin": bucket,
                    "tp": counts["tp"],
                    "fn": counts["fn"],
                    "recall": fmt(recall),
                }
            )

    proposed = read_proposed_metrics(args.proposed_metrics, args.proposed_negative_json, args.proposed_log)
    if proposed:
        summary_rows.insert(
            0,
            {
                "method": proposed["method"],
                "description": proposed["description"],
                "threshold_intensity": proposed["threshold_intensity"],
                "match_threshold_xy_m": proposed["match_threshold_xy_m"],
                "tp": proposed["tp"],
                "fp": proposed["fp"],
                "fn": proposed["fn"],
                "precision": fmt(proposed["precision"]),
                "recall": fmt(proposed["recall"]),
                "f1": fmt(proposed["f1"]),
                "negative_frame_count": proposed.get("negative_frame_count", ""),
                "negative_fp_frames": proposed.get("negative_fp_frames", ""),
                "negative_fp_frame_rate": fmt(proposed.get("negative_fp_frame_rate", "")),
                "negative_fp_total": proposed.get("negative_fp_total", ""),
                "mean_frame_ms": fmt(proposed.get("mean_frame_ms", "")),
                "p95_frame_ms": fmt(proposed.get("p95_frame_ms", "")),
                "max_frame_ms": fmt(proposed.get("max_frame_ms", "")),
            },
        )

    sensitivity_rows = []
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    sensitivity_methods = ["sorla_like_reflective_cluster", "single_frame_geometry"]
    for method in sensitivity_methods:
        for threshold in thresholds:
            detections_by_stamp = {}
            for stamp in all_stamps:
                item = clouds.get(stamp)
                detections_by_stamp[stamp] = (
                    run_baseline(item["points"], method, threshold, args.cluster_eps) if item is not None else []
                )
            result = evaluate_method(detections_by_stamp, positives, negatives, args.match_threshold_xy)
            sensitivity_rows.append(
                {
                    "method": method,
                    "threshold_intensity": threshold,
                    "match_threshold_xy_m": args.match_threshold_xy,
                    "tp": result["tp"],
                    "fp": result["fp"],
                    "fn": result["fn"],
                    "precision": fmt(result["precision"]),
                    "recall": fmt(result["recall"]),
                    "f1": fmt(result["f1"]),
                    "negative_fp_frames": result["negative_fp_frames"],
                    "negative_fp_frame_rate": fmt(result["negative_fp_frame_rate"]),
                    "negative_fp_total": result["negative_fp_total"],
                }
            )

    detection_fields = [
        "method",
        "stamp",
        "cloud_stamp",
        "stamp_delta_s",
        "det_index",
        "x",
        "y",
        "z",
        "n_points",
        "mean_intensity",
        "max_intensity",
        "xy_span",
        "xy_radius",
        "z_range",
        "std_xy",
        "aspect_ratio",
        "z_bands",
        "range_xy",
    ]
    summary_fields = [
        "method",
        "description",
        "threshold_intensity",
        "match_threshold_xy_m",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "negative_frame_count",
        "negative_fp_frames",
        "negative_fp_frame_rate",
        "negative_fp_total",
        "mean_frame_ms",
        "p95_frame_ms",
        "max_frame_ms",
    ]
    write_csv(out_dir / "baseline_detections.csv", detection_fields, detection_rows)
    write_csv(out_dir / "baseline_comparison_summary.csv", summary_fields, summary_rows)
    write_csv(out_dir / "baseline_distance_bins.csv", ["method", "range_bin", "tp", "fn", "recall"], bin_rows)
    write_csv(out_dir / "baseline_timing_summary.csv", ["method", "mean_ms", "median_ms", "p95_ms", "max_ms"], timing_rows)
    write_csv(
        out_dir / "threshold_sensitivity.csv",
        [
            "method",
            "threshold_intensity",
            "match_threshold_xy_m",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "negative_fp_frames",
            "negative_fp_frame_rate",
            "negative_fp_total",
        ],
        sensitivity_rows,
    )

    manifest = {
        "bag": os.path.abspath(args.bag),
        "ground_truth": os.path.abspath(args.ground_truth),
        "lidar_topic": args.lidar_topic,
        "labeled_frame_count": len(all_stamps),
        "positive_frame_count": len(positives),
        "positive_instance_count": sum(len(rows) for rows in positives.values()),
        "negative_frame_count": len(negatives),
        "missing_pointcloud_frames": missing,
        "frame_tolerance_s": args.frame_tolerance,
        "match_threshold_xy_m": args.match_threshold_xy,
        "cluster_eps_m": args.cluster_eps,
        "primary_threshold": args.primary_threshold,
        "thresholds": thresholds,
        "methods": [{"method": method, "description": desc} for method, desc in methods],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("Baseline evaluation finished:")
    print("  out:", out_dir)
    print("  labeled frames:", len(all_stamps))
    print("  missing pointcloud frames:", len(missing))
    print("  summary:", out_dir / "baseline_comparison_summary.csv")


if __name__ == "__main__":
    main()
