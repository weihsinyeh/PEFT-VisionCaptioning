#!/bin/bash

# Set the base directory
base_dir="/project/g/r13922043/hw3_output/P1_pred"

# Find the highest existing index
index=0
while [ -f "$base_dir/prediction_${index}.json" ]; do
    ((index++))
done

# Set the prediction file to the highest existing index
pred_file="$base_dir/prediction_${index}.json"

# Run the commands with the selected prediction file
python3 P1/finetune.py --pred_file "$pred_file"
python3 evaluate.py --pred_file "$pred_file" --images_root /project/g/r13922043/hw3_data/p1_data/images/val --annotation_file /project/g/r13922043/hw3_data/p1_data/val.json