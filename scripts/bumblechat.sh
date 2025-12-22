#!/bin/bash
CUDA_VISIBLE_DEVICES="0,1" python src/api.py \
    --yaml_config ./configs/inference/bumblechat.yaml \
    --model_path /data/wangxinhao/bumblecore/checkpoints/sft/bumblebee_1.5b_Instruct/checkpoint-1000 \
    --device_map cpu