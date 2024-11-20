#!/bin/bash

# Run the commands with the selected prediction file

# bash hw3_1.sh "/project/g/r13922043/hw3_data/p1_data/images/val" "/project/g/r13922043/hw3_output/P1_pred/pred_50.json"
# python3 P1/inference.py --images_dir /project/g/r13922043/hw3_data/p1_data/images/val --pred_file testP1.json
python3 P1/inference.py --images_dir $1 --pred_file $2

# Evalation the prediction file
# PB1
# CUDA_VISIBLE_DEVICES=1 python3 evaluate.py --pred_file test.json --images_root /project/g/r13922043/hw3_data/p1_data/images/val --annotation_file /project/g/r13922043/hw3_data/p1_data/val.json 