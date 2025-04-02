#!/bin/bash
# conda activate designcog

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4



video_name="cat4"
test_name="gracecat4"
prompt="A cat is standing in a broad living room on a wooden floor. The sunlight shine from the window and shine the cat's shadow and create a warm scene. The background is clear and bright."

# preprocess the input video
python src/preprocess.py --input_path="./input/${video_name}.mp4"

python src/IC-Light/video_gen_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./input/${video_name}_preprocessed.mp4" --output_path="./output/${test_name}icic.mp4" --prompt="${prompt}" --mix_bg
