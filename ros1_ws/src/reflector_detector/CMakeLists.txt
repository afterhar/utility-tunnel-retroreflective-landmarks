cmake_minimum_required(VERSION 3.0.2)
project(reflector_detector)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
)

add_message_files(
  FILES
  ReflectorObservation.msg
  ReflectorObservationArray.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/reflector_detector_v3_node.py
  scripts/reflector_sparse_detector.py
  scripts/reflector_tracker.py
  scripts/imu_deskew.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

