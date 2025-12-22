#!/bin/bash
CUDA_VISIBLE_DEVICES="0,1" python src/inference.py \
    --yaml_config ./configs/inference/chat.yaml \
    --model_path <your model path> \
    --device_map auto