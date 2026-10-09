# Runtime audit, 9 October 2026

The published main branch at audit start was commit `c0c7b84`.
That version had passed syntax and XML checks, but had not passed a ROS build
or runtime smoke test. A fresh build failed because package.xml omitted rospy
as a build dependency. The CMake install destinations also contained escaped
variable references. The evaluation guide used a nonexistent --xy-threshold
option; the actual option is --thresholds and --out specifies a directory.

These packaging issues were corrected locally. Escaped Markdown backticks
were removed, and the build instructions now source ROS before catkin_make.

## Tests performed

- ROS Noetic catkin build with Python 3: passed.
- Launch through the packaged reflector_v3.launch on an isolated ROS master:
  passed.
- Thirty synthetic vertical-column clouds: 29 output messages received,
  with 28 STABLE observations. All stable XY centers were within 0.05 m of
  the synthetic center (2 m, 1 m).

Run the smoke test after building and sourcing the workspace:

```bash
source /opt/ros/noetic/setup.bash
source ros1_ws/devel/setup.bash
python3 evaluation/scripts/smoke_test_ros.py
```

This test validates startup, generated ROS messages, frontend processing,
tracking, and stable observation publication. It does not reproduce the
paper's real-bag metrics, full localization experiments, sensor calibration,
or RViz graphical display.

## Completeness relative to the paper

The repository contains the observation frontend and selected evaluation and
labeling tools. It does not contain the downstream reflector localization
backend, covariance-update variants, tunnel_100m Gazebo world, reflector maps,
robot/controller dependencies, paired-trajectory experiment executors, clutter
world generators, or full experiment result inputs. These exist in the original
engineering workspace but were not packaged in this release. Consequently the
repository cannot independently reproduce all localization and synthetic-clutter
tables in the paper.
