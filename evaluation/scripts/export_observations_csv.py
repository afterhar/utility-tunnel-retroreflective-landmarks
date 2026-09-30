#!/usr/bin/env python3
"""Export reflector detector output bags to CSV files for paper experiments."""

import argparse
import csv
import json
import math
import os
import re
from collections import OrderedDict


STATE_NAMES = {
    0: "TENTATIVE",
    1: "STABLE",
    2: "DEGRADED",
    3: "LOST",
}


def stamp_to_sec(stamp):
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def vec_fields(value, names):
    row = {}
    for name in names:
        row[name] = getattr(value, name, "")
    return row


def parse_timing_text(text):
    if "[ReflectorDetector] timing" not in text and "V3 timing" not in text:
        return None
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)", text)
    parsed = OrderedDict()
    for key, raw_value in pairs:
        value = raw_value.rstrip(",")
        if value.endswith("ms"):
            value = value[:-2]
            if not key.endswith("_ms"):
                key = key + "_ms"
        if value in ("yes", "no"):
            parsed[key] = value
            continue
        try:
            if "." in value or "e" in value.lower():
                parsed[key] = float(value)
            else:
                parsed[key] = int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def export_observations(bag, obs_topic, out_path):
    fields = [
        "stamp",
        "frame_id",
        "topic",
        "obs_index",
        "tracker_id",
        "x",
        "y",
        "z",
        "range_xy",
        "range_3d",
        "confidence",
        "state",
        "state_name",
        "gate_mode",
        "stable_count",
        "missing_count",
        "gate_radius",
        "score_intensity",
        "score_shape",
        "score_temporal",
        "score_prediction",
        "association_cost",
    ]
    count = 0
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for topic, msg, _ in bag.read_messages(topics=[obs_topic]):
            stamp = stamp_to_sec(msg.header.stamp)
            frame_id = getattr(msg.header, "frame_id", "")
            for idx, obs in enumerate(getattr(msg, "observations", [])):
                pos = obs.pose.position
                range_xy = math.hypot(pos.x, pos.y)
                range_3d = math.sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z)
                state = int(getattr(obs, "state", -1))
                writer.writerow(
                    {
                        "stamp": "%.9f" % stamp,
                        "frame_id": frame_id,
                        "topic": topic,
                        "obs_index": idx,
                        "tracker_id": getattr(obs, "tracker_id", ""),
                        "x": "%.6f" % pos.x,
                        "y": "%.6f" % pos.y,
                        "z": "%.6f" % pos.z,
                        "range_xy": "%.6f" % range_xy,
                        "range_3d": "%.6f" % range_3d,
                        "confidence": "%.6f" % getattr(obs, "confidence", 0.0),
                        "state": state,
                        "state_name": STATE_NAMES.get(state, "UNKNOWN"),
                        "gate_mode": getattr(obs, "gate_mode", ""),
                        "stable_count": getattr(obs, "stable_count", ""),
                        "missing_count": getattr(obs, "missing_count", ""),
                        "gate_radius": "%.6f" % getattr(obs, "gate_radius", 0.0),
                        "score_intensity": "%.6f" % getattr(obs, "score_intensity", 0.0),
                        "score_shape": "%.6f" % getattr(obs, "score_shape", 0.0),
                        "score_temporal": "%.6f" % getattr(obs, "score_temporal", 0.0),
                        "score_prediction": "%.6f" % getattr(obs, "score_prediction", 0.0),
                        "association_cost": "%.6f" % getattr(obs, "association_cost", 0.0),
                    }
                )
                count += 1
    return count


def export_imu(bag, imu_topics, out_path):
    fields = [
        "stamp",
        "topic",
        "orientation_x",
        "orientation_y",
        "orientation_z",
        "orientation_w",
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
        "linear_acceleration_x",
        "linear_acceleration_y",
        "linear_acceleration_z",
    ]
    count = 0
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for topic, msg, _ in bag.read_messages(topics=imu_topics):
            stamp = stamp_to_sec(msg.header.stamp)
            row = {"stamp": "%.9f" % stamp, "topic": topic}
            row.update(vec_fields(msg.orientation, ["x", "y", "z", "w"]))
            row["orientation_x"] = row.pop("x")
            row["orientation_y"] = row.pop("y")
            row["orientation_z"] = row.pop("z")
            row["orientation_w"] = row.pop("w")
            row["angular_velocity_x"] = getattr(msg.angular_velocity, "x", "")
            row["angular_velocity_y"] = getattr(msg.angular_velocity, "y", "")
            row["angular_velocity_z"] = getattr(msg.angular_velocity, "z", "")
            row["linear_acceleration_x"] = getattr(msg.linear_acceleration, "x", "")
            row["linear_acceleration_y"] = getattr(msg.linear_acceleration, "y", "")
            row["linear_acceleration_z"] = getattr(msg.linear_acceleration, "z", "")
            writer.writerow(row)
            count += 1
    return count


def export_timing(bag, rosout_topic, out_path):
    rows = []
    timing_keys = OrderedDict()
    for topic, msg, t in bag.read_messages(topics=[rosout_topic]):
        text = getattr(msg, "msg", "")
        parsed = parse_timing_text(text)
        if parsed is None:
            continue
        stamp = stamp_to_sec(getattr(msg, "header", None).stamp) if hasattr(msg, "header") else stamp_to_sec(t)
        row = OrderedDict()
        row["stamp"] = "%.9f" % stamp
        row["topic"] = topic
        row["logger"] = getattr(msg, "name", "")
        row["level"] = getattr(msg, "level", "")
        for key, value in parsed.items():
            row[key] = value
            timing_keys[key] = True
        rows.append(row)

    fields = ["stamp", "topic", "logger", "level"] + list(timing_keys.keys())
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Detector output rosbag.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--obs-topic", default="/reflector_observations_detailed")
    parser.add_argument(
        "--imu-topics",
        default="/imu/data,/rs485_imu/data",
        help="Comma-separated IMU topics to export if present.",
    )
    parser.add_argument("--rosout-topic", default="/rosout")
    args = parser.parse_args()

    ensure_dir(args.out)

    try:
        import rosbag
    except ImportError as exc:
        raise SystemExit("Cannot import rosbag. Source your ROS environment first.") from exc

    imu_topics = [item.strip() for item in args.imu_topics.split(",") if item.strip()]
    summary_path = os.path.join(args.out, "summary.json")
    obs_path = os.path.join(args.out, "observations.csv")
    imu_path = os.path.join(args.out, "imu.csv")
    timing_path = os.path.join(args.out, "timing.csv")

    with rosbag.Bag(args.bag, "r") as bag:
        topics = bag.get_type_and_topic_info().topics
        obs_count = export_observations(bag, args.obs_topic, obs_path) if args.obs_topic in topics else 0
        imu_count = export_imu(bag, [topic for topic in imu_topics if topic in topics], imu_path)
        timing_count = export_timing(bag, args.rosout_topic, timing_path) if args.rosout_topic in topics else 0
        summary = {
            "bag": os.path.abspath(args.bag),
            "out": os.path.abspath(args.out),
            "start_time": bag.get_start_time(),
            "end_time": bag.get_end_time(),
            "duration": bag.get_end_time() - bag.get_start_time(),
            "topics": sorted(topics.keys()),
            "observation_rows": obs_count,
            "imu_rows": imu_count,
            "timing_rows": timing_count,
        }

    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("Export finished:")
    print("  observations:", obs_path, obs_count)
    print("  imu:", imu_path, imu_count)
    print("  timing:", timing_path, timing_count)
    print("  summary:", summary_path)


if __name__ == "__main__":
    main()
