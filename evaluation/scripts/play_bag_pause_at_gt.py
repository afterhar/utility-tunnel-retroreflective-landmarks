#!/usr/bin/env python3
"""Play a ROS bag and pause at ground-truth labeling timestamps.

This is an annotation helper. It publishes the selected bag topics and /clock
directly through rospy, then waits for user input at each timestamp read from a
GT template CSV. It does not write labels or change the bag.
"""

import argparse
import csv
import math
import os
import time


def parse_topic_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def stamp_to_sec(stamp):
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def msg_stamp_to_sec(msg, fallback_time):
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        try:
            return stamp_to_sec(stamp)
        except (AttributeError, TypeError, ValueError):
            pass
    return stamp_to_sec(fallback_time)


def load_targets(path, stamp_column, start_index, max_pauses):
    targets = []
    seen = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if stamp_column not in (reader.fieldnames or []):
            raise SystemExit("CSV does not contain stamp column: %s" % stamp_column)
        for row in reader:
            raw = (row.get(stamp_column) or "").strip()
            if not raw:
                continue
            try:
                stamp = float(raw)
            except ValueError:
                continue
            key = "%.9f" % stamp
            if key in seen:
                continue
            seen.add(key)
            targets.append(stamp)

    targets.sort()
    if start_index > 1:
        targets = targets[start_index - 1 :]
    if max_pauses > 0:
        targets = targets[:max_pauses]
    return targets


def format_sec(value):
    return "%.9f" % value


def wait_for_continue(index, total, target, current, topic):
    print("")
    print("=" * 72)
    print("Paused for GT labeling %d/%d" % (index, total))
    print("target stamp : %s" % format_sec(target))
    print("paused stamp : %s" % format_sec(current))
    print("pause topic  : %s" % topic)
    print("Fill the matching CSV row now. Press Enter to continue, or type q then Enter to quit.")
    try:
        answer = input("> ").strip().lower()
    except EOFError:
        print("")
        print("Input closed; stopping playback.")
        return False
    return answer not in ("q", "quit", "exit")


def advertise_bag_topics(topic_info, topics, queue_size):
    import roslib.message
    import rospy

    publishers = {}
    unresolved = []
    for topic in topics:
        msg_type = getattr(topic_info[topic], "msg_type", "")
        msg_class = roslib.message.get_message_class(msg_type)
        if msg_class is None:
            unresolved.append("%s (%s)" % (topic, msg_type))
            continue
        publishers[topic] = rospy.Publisher(topic, msg_class, queue_size=queue_size)
    return publishers, unresolved


def main():
    parser = argparse.ArgumentParser(
        description="Publish a rosbag and pause at timestamps from a GT CSV template."
    )
    parser.add_argument("--bag", required=True, help="Raw input bag.")
    parser.add_argument("--gt", required=True, help="Ground-truth template CSV containing stamp column.")
    parser.add_argument("--stamp-column", default="stamp")
    parser.add_argument(
        "--topics",
        default="/rslidar_points,/imu/data",
        help="Comma-separated bag topics to publish. Default: /rslidar_points,/imu/data",
    )
    parser.add_argument(
        "--pause-topic",
        default="/rslidar_points",
        help="Pause after publishing this topic at or after each target stamp.",
    )
    parser.add_argument("--rate", type=float, default=0.5, help="Playback rate. Default: 0.5")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Seconds allowed before a target when deciding to pause. Default: 0.03",
    )
    parser.add_argument("--start-index", type=int, default=1, help="1-based target index to start from.")
    parser.add_argument("--max-pauses", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--queue-size",
        type=int,
        default=10,
        help="Publisher queue size for replayed topics.",
    )
    parser.add_argument(
        "--rviz-topic",
        default="/reflector_detector/processed_input_cloud",
        help="PointCloud2 topic shown by the bundled RViz config.",
    )
    args = parser.parse_args()

    if args.rate <= 0.0:
        raise SystemExit("--rate must be > 0")
    if not os.path.isfile(args.bag):
        raise SystemExit("Bag not found: %s" % args.bag)
    if not os.path.isfile(args.gt):
        raise SystemExit("GT CSV not found: %s" % args.gt)

    try:
        import rosbag
        import rospy
        from rosgraph_msgs.msg import Clock
    except ImportError as exc:
        raise SystemExit(
            "Cannot import ROS Python modules. Run:\n"
            "  source /opt/ros/noetic/setup.bash\n"
            "  source <your_catkin_workspace>/devel/setup.bash"
        ) from exc

    topics = parse_topic_list(args.topics)
    if args.pause_topic not in topics:
        topics.append(args.pause_topic)
    targets = load_targets(args.gt, args.stamp_column, args.start_index, args.max_pauses)
    if not targets:
        raise SystemExit("No target stamps found in: %s" % args.gt)

    rospy.init_node("gt_pause_bag_player", anonymous=True)
    rospy.set_param("/use_sim_time", True)
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=10)

    with rosbag.Bag(args.bag, "r") as bag:
        topic_info = bag.get_type_and_topic_info().topics
        missing_topics = [topic for topic in topics if topic not in topic_info]
        if missing_topics:
            print("Warning: topics not found in bag: %s" % ", ".join(missing_topics))
        play_topics = [topic for topic in topics if topic in topic_info]
        if args.pause_topic not in play_topics:
            raise SystemExit("Pause topic not available in bag: %s" % args.pause_topic)

        publishers, unresolved_types = advertise_bag_topics(topic_info, play_topics, args.queue_size)
        time.sleep(0.5)

        print("Bag: %s" % os.path.abspath(args.bag))
        print("GT CSV: %s" % os.path.abspath(args.gt))
        print("Topics: %s" % ", ".join(play_topics))
        print("Pause topic: %s" % args.pause_topic)
        print("Targets: %d, starting at index %d" % (len(targets), args.start_index))
        print("Advertised replay topics: %s" % ", ".join(sorted(publishers.keys())))
        if unresolved_types:
            print("Warning: could not pre-advertise: %s" % ", ".join(unresolved_types))
        print("RViz default PointCloud2 topic: %s" % args.rviz_topic)
        print("Raw replay PointCloud2 topic: /rslidar_points")
        print("Press Enter to start publishing bag messages; RViz stays blank until the first cloud is published.")
        try:
            input("> ")
        except EOFError:
            print("")
            print("Input closed before playback started.")
            return

        target_index = 0
        last_time = None
        wall_last = None

        for topic, msg, bag_time in bag.read_messages(topics=play_topics):
            if rospy.is_shutdown():
                break

            current_sec = stamp_to_sec(bag_time)
            while target_index < len(targets) and targets[target_index] < current_sec - 5.0:
                print(
                    "Skipping target %d/%d before current bag time: %s"
                    % (target_index + 1, len(targets), format_sec(targets[target_index]))
                )
                target_index += 1
            if target_index >= len(targets):
                break

            if last_time is not None:
                dt = max(0.0, current_sec - last_time) / args.rate
                if wall_last is not None:
                    elapsed = time.time() - wall_last
                    remaining = dt - elapsed
                    if remaining > 0.0 and math.isfinite(remaining):
                        time.sleep(remaining)
            wall_last = time.time()
            last_time = current_sec

            if topic not in publishers:
                publishers[topic] = rospy.Publisher(topic, type(msg), queue_size=args.queue_size)
                time.sleep(0.05)

            clock_pub.publish(Clock(clock=bag_time))
            publishers[topic].publish(msg)

            if topic != args.pause_topic:
                continue

            pause_sec = msg_stamp_to_sec(msg, bag_time)
            target = targets[target_index]
            if pause_sec + args.tolerance >= target:
                keep_going = wait_for_continue(
                    target_index + 1,
                    len(targets),
                    target,
                    pause_sec,
                    topic,
                )
                target_index += 1
                wall_last = time.time()
                if not keep_going:
                    print("Stopped by user.")
                    return

        print("")
        print("Playback finished. Paused at %d/%d target stamps." % (target_index, len(targets)))


if __name__ == "__main__":
    main()
