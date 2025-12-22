#!/bin/bash
deepspeed --include localhost:0,1 src/train.py \
    --yaml_config ./configs/pretrain/pretrain_full.yaml \
    --model_name_or_path ./models/bumblebee \
    --dataset_path <your dataset path> \
    --output_dir ./checkpoints/pretrain/bumblebee_1.5b_base