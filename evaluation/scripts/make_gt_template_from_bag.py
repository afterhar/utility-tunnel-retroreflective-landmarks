#!/usr/bin/env python3
"""Create a manual GT labeling template directly from rosbag point-cloud stamps."""

import argparse
import bisect
import csv
import os


FIELDS = [
    "bag_name",
    "stamp",
    "gt_id",
    "x",
    "y",
    "z",
    "visibility",
    "range_bin",
    "angle_deg",
    "notes",
]


def stamp_to_sec(stamp):
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def msg_stamp_to_sec(msg, bag_time):
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        try:
            value = stamp_to_sec(stamp)
            if value > 0.0:
                return value
        except (AttributeError, TypeError, ValueError):
            pass
    return stamp_to_sec(bag_time)


def read_topic_stamps(bag_path, topic):
    import rosbag

    stamps = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, bag_time in bag.read_messages(topics=[topic]):
            stamps.append(msg_stamp_to_sec(msg, bag_time))
    return sorted(set(stamps))


def select_by_step(stamps, sample_step, start_time, end_time, max_rows):
    selected = []
    last = None
    for stamp in stamps:
        if stamp < start_time or stamp > end_time:
            continue
        if last is None or stamp - last >= sample_step:
            selected.append(stamp)
            last = stamp
            if max_rows > 0 and len(selected) >= max_rows:
                break
    return selected


def select_by_count(stamps, count, start_time, end_time):
    available = [stamp for stamp in stamps if start_time <= stamp <= end_time]
    if count <= 0 or count >= len(available):
        return available
    if count == 1:
        return [available[len(available) // 2]]

    span = end_time - start_time
    selected = []
    seen = set()
    for i in range(count):
        target = start_time + span * i / float(count - 1)
        idx = bisect.bisect_left(available, target)
        candidates = []
        if idx < len(available):
            candidates.append(available[idx])
        if idx > 0:
            candidates.append(available[idx - 1])
        if not candidates:
            continue
        stamp = min(candidates, key=lambda value: abs(value - target))
        key = "%.9f" % stamp
        if key not in seen:
            selected.append(stamp)
            seen.add(key)
    return selected


def range_bin_for_stamp(stamp, first_stamp, last_stamp):
    # Keep this as an annotation-progress bin, not target range.  It helps split
    # long bags into review chunks while preserving the evaluator's CSV schema.
    ratio = (stamp - first_stamp) / max(last_stamp - first_stamp, 1e-6)
    if ratio < 1.0 / 3.0:
        return "t0_t33"
    if ratio < 2.0 / 3.0:
        return "t33_t66"
    return "t66_t100"


def write_template(path, bag_name, stamps, first_stamp, last_stamp):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for stamp in stamps:
            writer.writerow(
                {
                    "bag_name": bag_name,
                    "stamp": "%.9f" % stamp,
                    "gt_id": "",
                    "x": "",
                    "y": "",
                    "z": "",
                    "visibility": "1",
                    "range_bin": range_bin_for_stamp(stamp, first_stamp, last_stamp),
                    "angle_deg": "",
                    "notes": "duplicate this row if multiple reflectors are visible",
                }
            )


def write_stamp_list(path, stamps):
    with open(path, "w") as handle:
        for index, stamp in enumerate(stamps, start=1):
            handle.write("%03d %.9f\n" % (index, stamp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topic", default="/rslidar_points")
    parser.add_argument("--bag-name", default="")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--sample-step", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--start-offset", type=float, default=2.0)
    parser.add_argument("--end-offset", type=float, default=2.0)
    parser.add_argument("--stamp-list-out", default="")
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        raise SystemExit("Bag not found: %s" % args.bag)
    if args.sample_step < 0.0:
        raise SystemExit("--sample-step must be >= 0")

    stamps = read_topic_stamps(args.bag, args.topic)
    if not stamps:
        raise SystemExit("No messages found on topic %s in %s" % (args.topic, args.bag))

    first_stamp = stamps[0]
    last_stamp = stamps[-1]
    start_time = first_stamp + max(args.start_offset, 0.0)
    end_time = last_stamp - max(args.end_offset, 0.0)
    if end_time <= start_time:
        raise SystemExit("Invalid time window after offsets")

    if args.sample_step > 0.0:
        selected = select_by_step(stamps, args.sample_step, start_time, end_time, args.max_rows)
    else:
        selected = select_by_count(stamps, args.count, start_time, end_time)
        if args.max_rows > 0:
            selected = selected[: args.max_rows]

    bag_name = args.bag_name or os.path.splitext(os.path.basename(args.bag))[0]
    write_template(args.out, bag_name, selected, first_stamp, last_stamp)

    stamp_list_out = args.stamp_list_out
    if not stamp_list_out:
        root, _ = os.path.splitext(args.out)
        stamp_list_out = root + "_stamps.txt"
    write_stamp_list(stamp_list_out, selected)

    print("GT template:", os.path.abspath(args.out))
    print("Stamp list :", os.path.abspath(stamp_list_out))
    print("Bag topic  :", args.topic)
    print("Bag window : %.9f -> %.9f" % (first_stamp, last_stamp))
    print("Label rows :", len(selected))
    if selected:
        print("First/last : %.9f -> %.9f" % (selected[0], selected[-1]))


if __name__ == "__main__":
    main()
