#!/bin/bash
# conda activate designcog

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1



video_name="man9"
image_name="beach3"
image_ext="jpg"
test_name="graceman9"
prompt="A man with glasses and a hat is talking on a beach near a sea with waves. The golden sun shines on the sea, creating a glistening and rippling effect. The background is clear and bright."

# preprocess the input video
python src/preprocess.py --input_path="./input/${video_name}.mp4"
python src/preprocess_image.py --input_path="./input/${image_name}.${image_ext}"

python src/IC-Light/video_relight_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./input/${image_name}_preprocessed.mp4" --output_path="./output/${test_name}ic.mp4" --prompt="${prompt}" --mix_bg
python src/IC-Light/video_edit_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icic.mp4" --prompt="${prompt}" --mix_bg --strength=0.5
