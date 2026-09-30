#!/usr/bin/env python3
"""Write RViz clicked reflector centers into a GT CSV template."""

import argparse
import csv
import os
import re
import sys
import tempfile
import threading


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


def load_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            clean = {field: row.get(field, "") for field in FIELDS}
            rows.append(clean)
    return rows


def atomic_write_csv(path, rows):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".gt_click_", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in sorted(rows, key=lambda item: (float(item["stamp"]), item.get("gt_id", ""))):
                writer.writerow({field: row.get(field, "") for field in FIELDS})
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def is_empty_target_row(row):
    return (
        row.get("visibility", "1") != "0"
        and not row.get("x")
        and not row.get("y")
        and not row.get("z")
    )


def can_accept_click(row):
    return not row.get("x") and not row.get("y") and not row.get("z")


class ClickGtWriter:
    def __init__(self, template, out, tolerance, frame_id, id_prefix):
        self.template = template
        self.out = out
        self.tolerance = tolerance
        self.frame_id = frame_id
        self.id_prefix = id_prefix
        self.rows = load_rows(out if os.path.exists(out) else template)
        self.lock = threading.Lock()
        self.clock_sec = None
        self.stamps = sorted({float(row["stamp"]) for row in self.rows if row.get("stamp")})
        if not self.stamps:
            raise RuntimeError("No stamp rows found in %s" % template)
        self.next_id = self._initial_next_id()

    def _initial_next_id(self):
        max_seen = 0
        pattern = re.compile(r"^%s(\d+)$" % re.escape(self.id_prefix))
        for row in self.rows:
            match = pattern.match(row.get("gt_id", ""))
            if match:
                max_seen = max(max_seen, int(match.group(1)))
        return max_seen + 1

    def _new_id(self):
        value = "%s%03d" % (self.id_prefix, self.next_id)
        self.next_id += 1
        return value

    def update_clock(self, stamp):
        with self.lock:
            self.clock_sec = stamp_to_sec(stamp)

    def _nearest_stamp(self, fallback_stamp=None):
        query = self.clock_sec if self.clock_sec is not None else fallback_stamp
        if query is None:
            return None, None
        nearest = min(self.stamps, key=lambda value: abs(value - query))
        return nearest, abs(nearest - query)

    def _rows_at_stamp(self, stamp):
        key = "%.9f" % stamp
        return [row for row in self.rows if row.get("stamp") == key]

    def _base_row_for_stamp(self, stamp):
        rows = self._rows_at_stamp(stamp)
        if rows:
            base = rows[0].copy()
        else:
            base = {
                "bag_name": "",
                "stamp": "%.9f" % stamp,
                "gt_id": "",
                "x": "",
                "y": "",
                "z": "",
                "visibility": "1",
                "range_bin": "",
                "angle_deg": "",
                "notes": "",
            }
        return base

    def write_click(self, point_msg):
        fallback_stamp = stamp_to_sec(point_msg.header.stamp) if point_msg.header.stamp else None
        with self.lock:
            nearest, delta = self._nearest_stamp(fallback_stamp)
            if nearest is None:
                print("No /clock time yet; wait until the pause player publishes a frame.", flush=True)
                return
            if delta is None or delta > self.tolerance:
                print(
                    "Clicked point ignored: nearest template stamp is %.3fs away "
                    "(tolerance %.3fs)." % (delta, self.tolerance),
                    flush=True,
                )
                return
            if self.frame_id and point_msg.header.frame_id and point_msg.header.frame_id != self.frame_id:
                print(
                    "Warning: clicked point frame_id is %s, expected %s."
                    % (point_msg.header.frame_id, self.frame_id),
                    flush=True,
                )

            rows_at_stamp = self._rows_at_stamp(nearest)
            target = None
            for row in rows_at_stamp:
                if can_accept_click(row):
                    target = row
                    break
            if target is None:
                target = self._base_row_for_stamp(nearest)
                target.update({"gt_id": "", "x": "", "y": "", "z": "", "visibility": "1"})
                self.rows.append(target)

            if not target.get("gt_id"):
                target["gt_id"] = self._new_id()
            target["x"] = "%.4f" % point_msg.point.x
            target["y"] = "%.4f" % point_msg.point.y
            target["z"] = "%.4f" % point_msg.point.z
            target["visibility"] = "1"
            target["notes"] = self._append_note(target.get("notes", ""), "clicked_in_rviz")
            atomic_write_csv(self.out, self.rows)
            print(
                "Saved click: stamp=%.9f gt_id=%s xyz=(%s, %s, %s)"
                % (nearest, target["gt_id"], target["x"], target["y"], target["z"]),
                flush=True,
            )

    def mark_no_visible(self):
        with self.lock:
            nearest, delta = self._nearest_stamp()
            if nearest is None:
                print("No /clock time yet; cannot mark no-visible.", flush=True)
                return
            if delta is None or delta > self.tolerance:
                print(
                    "No-visible command ignored: nearest template stamp is %.3fs away "
                    "(tolerance %.3fs)." % (delta, self.tolerance),
                    flush=True,
                )
                return
            rows_at_stamp = self._rows_at_stamp(nearest)
            if any(row.get("x") or row.get("y") or row.get("z") for row in rows_at_stamp):
                print("This stamp already has clicked coordinates; not overwriting them.", flush=True)
                return
            target = rows_at_stamp[0] if rows_at_stamp else self._base_row_for_stamp(nearest)
            if not rows_at_stamp:
                self.rows.append(target)
            target["gt_id"] = ""
            target["x"] = ""
            target["y"] = ""
            target["z"] = ""
            target["visibility"] = "0"
            target["notes"] = self._append_note(target.get("notes", ""), "no_visible_reflector")
            atomic_write_csv(self.out, self.rows)
            print("Marked no-visible: stamp=%.9f" % nearest, flush=True)

    @staticmethod
    def _append_note(old_note, new_note):
        old_note = old_note.strip()
        if not old_note or old_note == "duplicate this row if multiple reflectors are visible":
            return new_note
        if new_note in [item.strip() for item in old_note.split(";")]:
            return old_note
        return old_note + ";" + new_note


def stdin_loop(writer):
    print("Commands: n=mark current paused frame no-visible, q=quit writer.", flush=True)
    for raw in sys.stdin:
        cmd = raw.strip().lower()
        if cmd in ("q", "quit", "exit"):
            try:
                import rospy

                rospy.signal_shutdown("user quit")
            except Exception:
                pass
            return
        if cmd in ("n", "0", "none", "no"):
            writer.mark_no_visible()
        elif cmd:
            print("Unknown command: %s" % cmd, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, help="GT template CSV.")
    parser.add_argument("--out", required=True, help="Manual GT CSV to update.")
    parser.add_argument("--clicked-topic", default="/clicked_point")
    parser.add_argument("--clock-topic", default="/clock")
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--frame-id", default="rslidar")
    parser.add_argument("--id-prefix", default="R")
    parser.add_argument("--no-stdin", action="store_true")
    args = parser.parse_args()

    try:
        import rospy
        from geometry_msgs.msg import PointStamped
        from rosgraph_msgs.msg import Clock
    except ImportError as exc:
        raise SystemExit("Cannot import ROS Python modules. Source ROS first.") from exc

    writer = ClickGtWriter(args.template, args.out, args.tolerance, args.frame_id, args.id_prefix)
    if not os.path.exists(args.out):
        atomic_write_csv(args.out, writer.rows)

    rospy.init_node("rviz_click_gt_writer", anonymous=True)
    rospy.Subscriber(args.clock_topic, Clock, lambda msg: writer.update_clock(msg.clock), queue_size=10)
    rospy.Subscriber(args.clicked_topic, PointStamped, writer.write_click, queue_size=10)

    print("RViz click GT writer started.", flush=True)
    print("  template: %s" % os.path.abspath(args.template), flush=True)
    print("  out:      %s" % os.path.abspath(args.out), flush=True)
    print("  topic:    %s" % args.clicked_topic, flush=True)
    print("Click reflector centers with RViz Publish Point.", flush=True)

    if not args.no_stdin:
        thread = threading.Thread(target=stdin_loop, args=(writer,))
        thread.daemon = True
        thread.start()

    rospy.spin()


if __name__ == "__main__":
    main()
