# Source Release Audit

## Included

- The paper's Python observation module, sparse recovery module, temporal tracker, IMU deskew helper, and ROS message definitions.
- Evaluation, manual-labeling, and CSV export scripts used to reproduce observation-level metrics from user-provided data.
- A sanitized ROS Noetic launch file and paper reference configuration.

## Documentation and Comments

All four runtime Python modules have module-level descriptions. The front end documents its clustering dispatch and processing architecture; the sparse module documents its candidate representation and relaxed/near-range paths; the tracker documents its Kalman, assignment, confidence, and state-machine roles. Evaluation scripts expose command-line parameters through `argparse` and include concise module purposes. This is sufficient for source inspection without adding redundant line-by-line comments.

## Excluded Deliberately

- Legacy V1/V2 variants that were not part of the final paper implementation.
- `build`, `devel`, logs, editor artifacts, and machine-specific paths.
- Raw rosbags, maps, manual site assets, and proprietary annotations.
- Third-party Gazebo, Unitree, and robot-navigation source trees. Their licenses, dependencies, and operational configuration require independent distribution decisions.

## Remaining Limitations

The GitHub Actions check only validates Python syntax because a full ROS Noetic runtime is not available on the hosted runner. Reproduction of numerical results requires a compatible ROS Noetic environment, a valid PointCloud2 intensity field, and manually reviewed labels.

