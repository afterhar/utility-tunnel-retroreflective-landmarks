# Algorithm Architecture

The package implements the observation chain evaluated in the paper. It is intentionally separated from a robot-specific navigation system.

1. **Point-cloud input.** The node receives a LiDAR point cloud and optionally buffers odometry and IMU messages.
2. **High-return filtering.** Points outside the configured range, height band, or intensity threshold are discarded.
3. **Dense candidates.** Grid or cKDTree accelerated XY clustering groups the remaining points. A candidate must satisfy compact horizontal extent, vertical span, vertical-band count, intensity, and aspect-ratio constraints.
4. **Sparse recovery.** Unclaimed strong points are aggregated by an XY-radius search. This path recovers weakly sampled reflector columns while using stricter close-range constraints.
5. **Candidate fusion.** Dense and sparse candidates within the configured horizontal distance are merged to avoid duplicate observations.
6. **Temporal association.** A Kalman prediction and Hungarian assignment associate candidates with tracks. The association gate expands under angular motion when configured.
7. **State management.** A confidence exponential moving average and dense/sparse state conditions promote tracks to \`STABLE\`, retain short disruptions as \`DEGRADED\`, and remove lost tracks.
8. **Output.** Only eligible stable or degraded tracks are published as pose arrays, detailed observations, and optional RViz markers.

The main implementation is \`ros1_ws/src/reflector_detector/scripts/reflector_detector_v3_node.py\`. The sparse path and tracker are isolated in \`reflector_sparse_detector.py\` and \`reflector_tracker.py\`, respectively, to make their interfaces and unit-level inspection straightforward.

