#!/bin/bash
# conda activate designcog

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1



video_name="woman1"
test_name="gracewoman1"
prompt="A beautiful woman holding a glass is in a park outside with warm atmosphere."

# preprocess the input video
python src/preprocess.py --input_path="./input/${video_name}.mp4"

python src/IC-Light/video_gen_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./input/${video_name}_preprocessed.mp4" --output_path="./output/${test_name}icic.mp4" --prompt="${prompt}" --mix_bg