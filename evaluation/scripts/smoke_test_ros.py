#!/usr/bin/env python3
"""Launch the packaged detector and verify stable output from synthetic columns.

Source ROS Noetic and the built workspace before running. This exercises the
runtime pipeline, not accuracy on the paper's real-world or simulation datasets.
"""
import math
import os
import signal
import subprocess
import tempfile
import time

import rospy
from reflector_detector.msg import ReflectorObservationArray
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def main():
    # Use a separate ROS master to avoid disturbing a running robot stack.
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:11431'
    os.environ['ROS_HOSTNAME'] = '127.0.0.1'
    received = []
    with tempfile.TemporaryFile(mode='w+') as log:
        launch = subprocess.Popen(
            ['roslaunch', '-p', '11431', 'reflector_detector',
             'reflector_v3.launch', 'input_topic:=/smoke_points'],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            time.sleep(4)
            rospy.init_node('reflector_release_smoke', disable_signals=True)
            sub = rospy.Subscriber('/reflector_observations_detailed',
                                   ReflectorObservationArray, received.append)
            pub = rospy.Publisher('/smoke_points', PointCloud2, queue_size=1)
            deadline = time.monotonic() + 15
            while not pub.get_num_connections():
                if time.monotonic() > deadline or launch.poll() is not None:
                    raise RuntimeError('Detector did not subscribe to point clouds')
                time.sleep(0.1)
            fields = [PointField(name, offset, PointField.FLOAT32, 1)
                      for name, offset in [('x', 0), ('y', 4), ('z', 8), ('intensity', 12)]]
            points = [(2 + 0.015 * math.cos(a), 1 + 0.015 * math.sin(a), z, 240)
                      for z in [-0.2, -0.1, 0, 0.1, 0.2]
                      for a in [0, math.pi / 2, math.pi, 3 * math.pi / 2]]
            for _ in range(30):
                pub.publish(point_cloud2.create_cloud(
                    Header(stamp=rospy.Time.now(), frame_id='rslidar'), fields, points))
                time.sleep(0.1)
            stable = [o for msg in received for o in msg.observations if o.state == 1]
            assert stable, 'No STABLE observations received'
            assert all(math.hypot(o.pose.position.x - 2, o.pose.position.y - 1) < 0.05
                       for o in stable), 'Synthetic center error exceeds 0.05 m'
            print('PASS: %d messages, %d stable observations' % (len(received), len(stable)))
        except Exception:
            log.seek(0)
            print(log.read())
            raise
        finally:
            rospy.signal_shutdown('Smoke test complete')
            if launch.poll() is None:
                os.killpg(launch.pid, signal.SIGINT)
                try:
                    launch.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(launch.pid, signal.SIGKILL)
                    launch.wait()


if __name__ == '__main__':
    main()
