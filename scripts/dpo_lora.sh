#!/bin/bash
deepspeed --include localhost:0,1 src/train.py \
    --yaml_config ./configs/dpo/dpo_lora.yaml \
    --model_name_or_path <your model path> \
    --dataset_path <your dataset path> \
    --output_dir ./checkpoints/dpo/bumblebee_1.5b_dpo_lora