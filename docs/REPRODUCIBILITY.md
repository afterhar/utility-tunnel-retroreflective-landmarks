# Reproducibility Guide

## Inputs

The ROS node requires a PointCloud2 topic containing \`x\`, \`y\`, \`z\`, and \`intensity\` fields. Odometry and IMU topics are subscribed to by default, although deskew remains disabled unless timing and extrinsic calibration have been verified.

The evaluation scripts use manually reviewed CSV labels. A label row contains at least:

\`\`\`text
bag_name,stamp,gt_id,x,y,z,visibility
example.bag,1777195652.729055,R001,1.65,1.35,0.0,1
example.bag,1777195666.329090,,,,,0
\`\`\`

Use \`visibility=1\` for a visible reflector and \`visibility=0\` for a reviewed no-target frame. A timestamp may have multiple positive rows when multiple reflectors are visible.

## Generate Detector Outputs

Run the node, replay a compatible rosbag, and record the detailed observation topic:

\`\`\`bash
rosbag record -O detector_output.bag /reflector_observations_detailed /rosout
python3 evaluation/scripts/export_observations_csv.py \
  --bag detector_output.bag \
  --out output_csv
\`\`\`

Run \`python3 evaluation/scripts/export_observations_csv.py --help\` to view topic and export options for the installed version.

## Evaluate a Method Output

\`\`\`bash
python3 evaluation/scripts/evaluate_detection.py \
  --detections output_csv/observations.csv \
  --ground-truth labels.csv \
  --xy-threshold 0.15 \
  --out metrics.json

python3 evaluation/scripts/evaluate_negative_fp.py \
  --detections output_csv/observations.csv \
  --negative-frames labels.csv \
  --out negative_fp_metrics
\`\`\`

The raw-bag baseline script samples the point cloud nearest each labeled timestamp and compares deterministic high-intensity clustering variants on the same label set:

\`\`\`bash
python3 evaluation/scripts/evaluate_reflector_baselines_from_bag.py --help
\`\`\`

The exact bag paths, map assets, and full raw recordings from the controlled industrial environment are intentionally not embedded in commands or configuration files.

