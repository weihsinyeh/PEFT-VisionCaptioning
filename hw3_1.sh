#!/bin/bash

# Run the commands with the selected prediction file

# bash hw3_1.sh "/project/g/r13922043/hw3_data/p1_data/images/val" "/project/g/r13922043/hw3_output/P1_pred/pred_50.json"
# python3 P1/inference.py --images_dir "/project/g/r13922043/hw3_data/p1_data/images/val" --pred_file "/project/g/r13922043/hw3_output/P1_pred/pred_50.json"
echo $1
echo $2
python3 P1/inference.py --images_dir $1 --pred_file $2