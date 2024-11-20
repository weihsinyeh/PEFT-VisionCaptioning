#!/bin/bash

# TODO - run your inference Python3 code

# bash hw3_2.sh $1 $2 $3
# $1: path to the folder containing test images (e.g. hw3/p2_data/images/test/)
# $2: path to the output json file (e.g. hw3/output_p2/pred.json)
# $3: path to the decoder weights (e.g. hw3/p2_data/decoder_model.bin)
# (This means that you don’t need to upload decoder_model.bin)
# Usage example
# bash hw3_2.sh /project/g/r13922043/hw3_data/p2_data/images/val ./PB2_test.json ./decoder_model.bin
python3 evaluationPB2.py --valid_images_dir $1 --pred_file $2 --decoder $3
