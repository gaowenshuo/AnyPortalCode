#!/bin/bash
# conda activate designcog

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4



video_name="woman6"
image_name="beach3"
image_ext="jpg"
test_name="gracewoman6"
prompt="A black woman with curly hair in a red suit and sweater holding a computer is smiling in a cyberpunk city street with neon lights. The background is clear and bright."

# preprocess the input video
python src/preprocess.py --input_path="./input/${video_name}.mp4"

python src/IC-Light/video_gen_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./input/${video_name}_preprocessed.mp4" --output_path="./output/${test_name}icic.mp4" --prompt="${prompt}" --mix_bg