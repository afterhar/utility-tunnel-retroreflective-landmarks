#!/usr/bin/env python3
"""Compute first-pass detection metrics for reflector landmark experiments."""

import argparse
import csv
import math
import os
from collections import defaultdict


def parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_detections(path, states, min_confidence):
    rows = []
    allowed = set(states)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            state_name = row.get("state_name", "")
            if allowed and state_name not in allowed:
                continue
            confidence = parse_float(row.get("confidence"), 0.0)
            if confidence < min_confidence:
                continue
            stamp = parse_float(row.get("stamp"))
            x = parse_float(row.get("x"))
            y = parse_float(row.get("y"))
            z = parse_float(row.get("z"), 0.0)
            if stamp is None or x is None or y is None:
                continue
            rows.append(
                {
                    "stamp": stamp,
                    "tracker_id": row.get("tracker_id", ""),
                    "x": x,
                    "y": y,
                    "z": z,
                    "confidence": confidence,
                    "state_name": state_name,
                }
            )
    return rows


def load_ground_truth(path):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            visibility = row.get("visibility", "1").strip()
            if visibility in ("0", "false", "False", "no"):
                continue
            stamp = parse_float(row.get("stamp"))
            x = parse_float(row.get("x"))
            y = parse_float(row.get("y"))
            z = parse_float(row.get("z"), 0.0)
            if stamp is None or x is None or y is None:
                continue
            range_3d = math.sqrt(x * x + y * y + z * z)
            rows.append(
                {
                    "stamp": stamp,
                    "gt_id": row.get("gt_id", ""),
                    "x": x,
                    "y": y,
                    "z": z,
                    "range_3d": range_3d,
                }
            )
    return rows


def load_rotation(path):
    if not path:
        return []
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamp = parse_float(row.get("stamp"))
            value = parse_float(row.get("rotation_rate"))
            if value is None:
                value = parse_float(row.get("angular_velocity_z"))
            if stamp is None or value is None:
                continue
            rows.append((stamp, abs(value)))
    rows.sort()
    return rows


def nearest_rotation(rotation_rows, stamp):
    if not rotation_rows:
        return None
    best = min(rotation_rows, key=lambda item: abs(item[0] - stamp))
    return best[1]


def distance(a, b, mode="xyz"):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    if mode == "xy":
        return math.hypot(dx, dy)
    dz = a["z"] - b["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def group_by_stamp(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["stamp"]].append(row)
    return grouped


def candidate_detections(detections, stamp, tolerance):
    return [det for det in detections if abs(det["stamp"] - stamp) <= tolerance]


def match_frame(gt_rows, det_rows, threshold, distance_mode):
    pairs = []
    for gi, gt in enumerate(gt_rows):
        for di, det in enumerate(det_rows):
            dist = distance(gt, det, distance_mode)
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
        matches.append((gi, di, dist))
    tp = len(matches)
    fp = max(0, len(det_rows) - len(used_det))
    fn = max(0, len(gt_rows) - len(used_gt))
    return tp, fp, fn, matches


def metrics(tp, fp, fn):
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def bin_name(range_3d):
    if range_3d < 2.0:
        return "0.9-2m"
    if range_3d < 4.0:
        return "2-4m"
    return ">4m"


def evaluate(detections, ground_truth, thresholds, time_tolerance, rotation_rows, rotation_threshold, distance_mode):
    gt_by_stamp = group_by_stamp(ground_truth)
    results = []
    bin_results = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    rotation_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "frames": 0, "max_miss_streak": 0})

    for threshold in thresholds:
        total_tp = total_fp = total_fn = 0
        miss_streak = 0
        max_miss_streak = 0
        for stamp, gt_rows in sorted(gt_by_stamp.items()):
            det_rows = candidate_detections(detections, stamp, time_tolerance)
            tp, fp, fn, matches = match_frame(gt_rows, det_rows, threshold, distance_mode)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            matched_gt = {gi for gi, _, _ in matches}
            for gi, gt in enumerate(gt_rows):
                bucket = bin_name(gt["range_3d"])
                if gi in matched_gt:
                    bin_results[threshold][bucket]["tp"] += 1
                else:
                    bin_results[threshold][bucket]["fn"] += 1

            rot = nearest_rotation(rotation_rows, stamp)
            if rot is not None and rot >= rotation_threshold:
                rr = rotation_results[threshold]
                rr["frames"] += 1
                rr["tp"] += tp
                rr["fp"] += fp
                rr["fn"] += fn
                if fn > 0:
                    miss_streak += 1
                    max_miss_streak = max(max_miss_streak, miss_streak)
                else:
                    miss_streak = 0

        precision, recall, f1 = metrics(total_tp, total_fp, total_fn)
        results.append(
            {
                "threshold": threshold,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        rotation_results[threshold]["max_miss_streak"] = max_miss_streak
    return results, bin_results, rotation_results


def write_summary(path, results):
    fields = ["threshold", "tp", "fp", "fn", "precision", "recall", "f1"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            out = dict(row)
            for key in ("precision", "recall", "f1"):
                out[key] = "%.6f" % out[key]
            writer.writerow(out)


def write_bins(path, bin_results):
    fields = ["threshold", "range_bin", "tp", "fn", "recall"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for threshold, buckets in sorted(bin_results.items()):
            for bucket, counts in sorted(buckets.items()):
                recall = counts["tp"] / float(counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
                writer.writerow(
                    {
                        "threshold": threshold,
                        "range_bin": bucket,
                        "tp": counts["tp"],
                        "fn": counts["fn"],
                        "recall": "%.6f" % recall,
                    }
                )


def write_rotation(path, rotation_results):
    fields = ["threshold", "frames", "tp", "fp", "fn", "precision", "recall", "f1", "max_miss_streak"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for threshold, counts in sorted(rotation_results.items()):
            precision, recall, f1 = metrics(counts["tp"], counts["fp"], counts["fn"])
            writer.writerow(
                {
                    "threshold": threshold,
                    "frames": counts["frames"],
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "precision": "%.6f" % precision,
                    "recall": "%.6f" % recall,
                    "f1": "%.6f" % f1,
                    "max_miss_streak": counts["max_miss_streak"],
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="observations.csv")
    parser.add_argument("--ground-truth", required=True, help="manual ground-truth CSV")
    parser.add_argument("--rotation-csv", default="", help="timing.csv or imu.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--thresholds", default="0.15,0.25,0.35")
    parser.add_argument("--time-tolerance", type=float, default=0.05)
    parser.add_argument("--states", default="STABLE,DEGRADED")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--rotation-threshold", type=float, default=0.35)
    parser.add_argument(
        "--distance-mode",
        choices=("xyz", "xy"),
        default="xyz",
        help="Use xyz distance by default. Use xy when GT z is not annotated.",
    )
    args = parser.parse_args()

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    states = [item.strip() for item in args.states.split(",") if item.strip()]
    detections = load_detections(args.detections, states, args.min_confidence)
    ground_truth = load_ground_truth(args.ground_truth)
    rotation_rows = load_rotation(args.rotation_csv)

    os.makedirs(args.out, exist_ok=True)
    results, bin_results, rotation_results = evaluate(
        detections,
        ground_truth,
        thresholds,
        args.time_tolerance,
        rotation_rows,
        args.rotation_threshold,
        args.distance_mode,
    )

    write_summary(os.path.join(args.out, "metrics_summary.csv"), results)
    write_bins(os.path.join(args.out, "distance_bins.csv"), bin_results)
    write_rotation(os.path.join(args.out, "rotation_metrics.csv"), rotation_results)

    print("Evaluation finished:")
    print("  detections:", len(detections))
    print("  ground_truth:", len(ground_truth))
    print("  out:", args.out)


if __name__ == "__main__":
    main()
