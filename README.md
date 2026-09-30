# Stable Retroreflective-Landmark Observation for Utility-Tunnel Robots

This repository contains the ROS Noetic reference implementation used for the paper *Stable Observation Generation of Retroreflective Landmarks for Utility-Tunnel Inspection Robots*.

It provides the proposed observation module: intensity and spatial filtering, dense and sparse reflector-candidate generation, geometric verification, candidate fusion, and temporal tracking. The output is a set of stable reflector observations suitable for association with a pre-built reflector map.

## Scope

The paper's central contribution is stable landmark observation generation in repetitive utility-tunnel environments. The code publishes reflector observations and RViz markers; it does not distribute a full robot navigation stack, a proprietary reflector map, raw rosbags, or third-party robot and Gazebo assets.

## Repository Layout

\`\`\`text
ros1_ws/src/reflector_detector/  ROS Noetic package and message definitions
evaluation/scripts/              CSV export, manual-labeling, and metric tools
docs/                            Architecture, data policy, audit, and reproduction notes
\`\`\`

## Requirements

The runtime was developed for Ubuntu 20.04, ROS Noetic, and Python 3. It requires ROS packages \`rospy\`, \`sensor_msgs\`, \`nav_msgs\`, \`geometry_msgs\`, \`visualization_msgs\`, and \`message_generation\`, plus Python packages NumPy, SciPy, and FilterPy.

\`\`\`bash
sudo apt-get install ros-noetic-desktop-full python3-numpy python3-scipy
python3 -m pip install --user -r requirements.txt
\`\`\`

## Build

\`\`\`bash
git clone https://github.com/afterhar/utility-tunnel-retroreflective-landmarks.git
cd utility-tunnel-retroreflective-landmarks/ros1_ws
catkin_make
source /opt/ros/noetic/setup.bash
source devel/setup.bash
\`\`\`

## Run the Observation Module

The supplied configuration reflects the experimental parameterization reported in the paper, including the common intensity threshold of \`140\`. It is an example configuration for a Livox Mid-360 deployment and must be re-tuned after changing sensor, reflector material, mounting pose, or environment.

\`\`\`bash
roslaunch reflector_detector reflector_v3.launch \
  input_topic:=/rslidar_points \
  imu_topic:=/imu/data \
  odom_topic:=/Odometry \
  frame_id:=rslidar
\`\`\`

In another terminal, replay a compatible bag if desired:

\`\`\`bash
source /opt/ros/noetic/setup.bash
rosbag play --clock /path/to/your_recording.bag
\`\`\`

Inputs are \`sensor_msgs/PointCloud2\`, \`sensor_msgs/Imu\`, and \`nav_msgs/Odometry\`. Principal outputs are:

| Topic | Type | Purpose |
| --- | --- | --- |
| \`/reflector_observations\` | \`geometry_msgs/PoseArray\` | Stable and degraded reflector positions |
| \`/reflector_observations_detailed\` | \`reflector_detector/ReflectorObservationArray\` | Positions, confidence, state, and tracking diagnostics |
| \`/reflector_markers\` | \`visualization_msgs/MarkerArray\` | RViz visualization |
| \`/filtered_pointcloud\` | \`sensor_msgs/PointCloud2\` | Strong points retained by the front end |

Use RViz to add a \`MarkerArray\` display for \`/reflector_markers\` and \`PointCloud2\` displays for the input and filtered clouds.

## Evaluation Tools

The scripts in \`evaluation/scripts\` operate on user-provided bags and CSV labels. They create no hidden data and accept paths as command-line arguments. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the label schema and representative commands.

## Data and Reproducibility

Raw recordings, maps, and industrial-site assets are not bundled because they include large files and controlled-site information. The paper's tabulated data and audit records are provided as supplementary material. Additional raw data may be requested from the corresponding author, subject to data-management approval. See [docs/DATA_POLICY.md](docs/DATA_POLICY.md).

## License and Citation

The code in this repository is released under the [BSD 3-Clause License](LICENSE). Please cite the associated paper when using this implementation; the citation record is in [CITATION.cff](CITATION.cff).

