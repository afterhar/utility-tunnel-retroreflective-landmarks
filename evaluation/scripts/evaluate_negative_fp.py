#!/usr/bin/env python3
"""Evaluate false positives on manually labeled no-target frames."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_negative_frames(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            visibility = (row.get("visibility") or "").strip().lower()
            if visibility not in ("0", "false", "no"):
                continue
            stamp = parse_float(row.get("stamp"))
            if stamp is None:
                continue
            rows.append(
                {
                    "stamp": stamp,
                    "bag_name": row.get("bag_name", ""),
                    "relative_time_s": row.get("relative_time_s", ""),
                    "negative_type": row.get("negative_type", ""),
                    "label_source": row.get("label_source", ""),
                    "notes": row.get("notes", ""),
                }
            )
    rows.sort(key=lambda row: row["stamp"])
    return rows


def load_detections(path, states, min_confidence):
    allowed = set(states)
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            state_name = row.get("state_name", "")
            if allowed and state_name not in allowed:
                continue
            confidence = parse_float(row.get("confidence"), 0.0)
            if confidence < min_confidence:
                continue
            stamp = parse_float(row.get("stamp"))
            if stamp is None:
                continue
            rows.append(
                {
                    "stamp": stamp,
                    "tracker_id": row.get("tracker_id", ""),
                    "x": row.get("x", ""),
                    "y": row.get("y", ""),
                    "z": row.get("z", ""),
                    "range_xy": row.get("range_xy", ""),
                    "confidence": "%.6f" % confidence,
                    "state_name": state_name,
                    "gate_mode": row.get("gate_mode", ""),
                }
            )
    rows.sort(key=lambda row: row["stamp"])
    return rows


def detections_near(detections, stamp, tolerance):
    return [row for row in detections if abs(row["stamp"] - stamp) <= tolerance]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="Detector observations.csv.")
    parser.add_argument("--negative-frames", required=True, help="CSV with visibility=0 no-target frames.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--time-tolerance", type=float, default=0.05)
    parser.add_argument("--states", default="STABLE,DEGRADED")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = [item.strip() for item in args.states.split(",") if item.strip()]
    negative_frames = load_negative_frames(args.negative_frames)
    detections = load_detections(args.detections, states, args.min_confidence)

    detail_rows = []
    frame_rows = []
    state_counter = Counter()

    for frame in negative_frames:
        near = detections_near(detections, frame["stamp"], args.time_tolerance)
        frame_rows.append(
            {
                "stamp": "%.9f" % frame["stamp"],
                "relative_time_s": frame["relative_time_s"],
                "negative_type": frame["negative_type"],
                "label_source": frame["label_source"],
                "fp_count": len(near),
                "has_fp": 1 if near else 0,
            }
        )
        for det in near:
            state_counter[det["state_name"]] += 1
            detail_rows.append(
                {
                    "negative_stamp": "%.9f" % frame["stamp"],
                    "detection_stamp": "%.9f" % det["stamp"],
                    "time_delta_s": "%.6f" % (det["stamp"] - frame["stamp"]),
                    "tracker_id": det["tracker_id"],
                    "x": det["x"],
                    "y": det["y"],
                    "z": det["z"],
                    "range_xy": det["range_xy"],
                    "confidence": det["confidence"],
                    "state_name": det["state_name"],
                    "gate_mode": det["gate_mode"],
                    "negative_type": frame["negative_type"],
                    "label_source": frame["label_source"],
                }
            )

    frame_count = len(negative_frames)
    total_fp = len(detail_rows)
    frames_with_fp = sum(row["has_fp"] for row in frame_rows)
    summary = {
        "detections": str(Path(args.detections).resolve()),
        "negative_frames": str(Path(args.negative_frames).resolve()),
        "states": states,
        "min_confidence": args.min_confidence,
        "time_tolerance_s": args.time_tolerance,
        "negative_frame_count": frame_count,
        "frames_with_false_positive": frames_with_fp,
        "frames_without_false_positive": frame_count - frames_with_fp,
        "false_positive_frame_rate": frames_with_fp / frame_count if frame_count else 0.0,
        "total_false_positive_detections": total_fp,
        "false_positive_detections_per_frame": total_fp / frame_count if frame_count else 0.0,
        "max_false_positive_detections_in_one_frame": max((row["fp_count"] for row in frame_rows), default=0),
        "false_positive_count_by_state": dict(state_counter),
    }

    summary_csv = out_dir / "negative_fp_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value})

    frame_csv = out_dir / "negative_fp_by_frame.csv"
    with frame_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["stamp", "relative_time_s", "negative_type", "label_source", "fp_count", "has_fp"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    details_csv = out_dir / "negative_fp_details.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "negative_stamp",
            "detection_stamp",
            "time_delta_s",
            "tracker_id",
            "x",
            "y",
            "z",
            "range_xy",
            "confidence",
            "state_name",
            "gate_mode",
            "negative_type",
            "label_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    with (out_dir / "negative_fp_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Negative FP evaluation finished:")
    print("  negative frames:", frame_count)
    print("  frames with false positives:", frames_with_fp)
    print("  total false-positive detections:", total_fp)
    print("  out:", out_dir)


if __name__ == "__main__":
    main()
