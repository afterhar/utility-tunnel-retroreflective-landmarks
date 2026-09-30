[English](README.md) | [**简体中文**](README.zh-CN.md)

# 面向综合管廊巡检机器人的稳定反光标志观测

本仓库提供论文 *Stable Observation Generation of Retroreflective Landmarks for Utility-Tunnel Inspection Robots* 所对应的 ROS Noetic 参考实现。

工程实现包含所提出的稳定观测模块：反射强度与空间范围筛选、稠密和稀疏反光候选生成、柱状几何确认、候选融合与多帧时序跟踪。模块输出稳定的反光标志观测，可进一步与预构建的反光标志地图进行关联。

## 适用范围

论文的核心贡献是面向重复结构综合管廊环境的稳定地标观测生成。本仓库发布反光标志观测和 RViz 可视化代码；不发布完整机器人导航栈、专有反光标志地图、原始 rosbag，以及第三方机器人和 Gazebo 资产。

## 目录结构

\`\`\`text
ros1_ws/src/reflector_detector/  ROS Noetic 功能包与消息定义
evaluation/scripts/              CSV 导出、人工标注和评价工具
docs/                            算法结构、数据边界、代码审查和复现实验说明
\`\`\`

## 运行环境

开发环境为 Ubuntu 20.04、ROS Noetic 和 Python 3。运行时需要 ROS 包 \`rospy\`、\`sensor_msgs\`、\`nav_msgs\`、\`geometry_msgs\`、\`visualization_msgs\`、\`message_generation\`，以及 NumPy、SciPy、FilterPy。

\`\`\`bash
sudo apt-get install ros-noetic-desktop-full python3-numpy python3-scipy
python3 -m pip install --user -r requirements.txt
\`\`\`

## 编译

\`\`\`bash
git clone https://github.com/afterhar/utility-tunnel-retroreflective-landmarks.git
cd utility-tunnel-retroreflective-landmarks/ros1_ws
catkin_make
source /opt/ros/noetic/setup.bash
source devel/setup.bash
\`\`\`

## 启动稳定观测模块

随仓库提供的配置对应论文中的实验参数，其中共同强度阈值为 \`140\`。该配置是 Livox Mid-360 部署示例；更换雷达、反光材料、传感器安装位姿或场景后，必须重新标定和调整参数。

\`\`\`bash
roslaunch reflector_detector reflector_v3.launch \
  input_topic:=/rslidar_points \
  imu_topic:=/imu/data \
  odom_topic:=/Odometry \
  frame_id:=rslidar
\`\`\`

需要回放兼容 rosbag 时，在另一终端执行：

\`\`\`bash
source /opt/ros/noetic/setup.bash
rosbag play --clock /path/to/your_recording.bag
\`\`\`

输入为 \`sensor_msgs/PointCloud2\`、\`sensor_msgs/Imu\` 和 \`nav_msgs/Odometry\`。主要输出如下：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| \`/reflector_observations\` | \`geometry_msgs/PoseArray\` | 稳定或退化状态下的反光标志位置 |
| \`/reflector_observations_detailed\` | \`reflector_detector/ReflectorObservationArray\` | 位置、置信度、状态和跟踪诊断信息 |
| \`/reflector_markers\` | \`visualization_msgs/MarkerArray\` | RViz 可视化 |
| \`/filtered_pointcloud\` | \`sensor_msgs/PointCloud2\` | 前端保留的高反射强度点云 |

在 RViz 中添加 \`MarkerArray\` 并选择 \`/reflector_markers\`，再添加输入点云和 \`/filtered_pointcloud\` 的 \`PointCloud2\` 显示即可查看结果。

## 评价工具

\`evaluation/scripts\` 中的工具针对用户自行提供的 bag 和 CSV 标注运行，不依赖隐藏数据或固定本机路径。人工标注格式与典型评价命令见 [复现实验说明](docs/REPRODUCIBILITY.md)。

## 数据与可复现性

原始录包、地图、现场图片和机器人专用 Gazebo 资产未纳入本仓库，其中部分属于受控地下管廊场景资料。论文的表格结果、人工标注审计记录和结果生成说明见补充材料。原始数据可按合理请求向通讯作者获取，并遵守现场与数据管理要求。详情见 [数据政策](docs/DATA_POLICY.md)。

## 许可证与引用

本仓库代码采用 [BSD 3-Clause License](LICENSE)。使用本实现时，请引用对应论文；软件引文信息见 [CITATION.cff](CITATION.cff)。

