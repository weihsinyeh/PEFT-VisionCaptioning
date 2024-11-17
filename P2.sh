# /project/g/r13922043/hw3_output/P2_pred
# epoch_num
epoch_num=$1
# convert the filename to the id
python3 P2/convert.py --pred_file /project/g/r13922043/hw3_output/P2_pred/Epoch_${epoch_num}.json --annotation /project/g/r13922043/hw3_data/p2_data/val.json

CUDA_VISIBLE_DEVICES=1 python3 evaluate.py --pred_file /project/g/r13922043/hw3_output/P2_pred/update_Epoch_${epoch_num}.json --images_root /project/g/r13922043/hw3_data/p2_data/images/val --annotation_file /project/g/r13922043/hw3_data/p2_data/val.json