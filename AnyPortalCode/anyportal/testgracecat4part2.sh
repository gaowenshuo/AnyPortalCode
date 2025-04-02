#!/bin/bash
# conda activate designcog

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4



video_name="cat4"
test_name="gracecat4"
prompt="A cat is standing in a broad living room on a wooden floor. The sunlight shine from the window and shine the cat's shadow and create a warm scene. The background is clear and bright."

python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}bg.mp4" --input_refined_path="./output/${test_name}bg.mp4" --input_control_path="./input/${video_name}_preprocessed.mp4" --output_path="./output/${test_name}bg.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=49 --forward_steps=1

# detail recovery
python src/getmask.py --input_path="./output/${test_name}bg" --output_path="./output/${test_name}bg_mask"
python src/ProPainter/inference_propainter.py --video="./output/${test_name}bg.mp4" --mask="./output/${test_name}bg_mask" --output="./output/${test_name}bg_inpaint" --height=480 --width=720 --fp16 --save_frames --mask_dilation=1
python src/detailrecover.py --input_source_path="./input/${video_name}_preprocessed" --input_target_path="./output/${test_name}bg" --output_path="./output/${test_name}bg_refined.mp4" --inpaint_path="./output/${test_name}bg_inpaint/${test_name}bg/frames"

python src/IC-Light/video_relight_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./output/${test_name}bg_inpaint/${test_name}bg/inpaint_out.mp4" --output_path="./output/${test_name}ic.mp4" --prompt="${prompt}" --mix_bg

python src/IC-Light/video_edit_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icic.mp4" --prompt="${prompt}" --mix_bg --strength=0.5

# renoise
python src/getmask.py --input_path="./input/${video_name}_preprocessed" --output_path="./output/${test_name}_mask"
echo "Processing step 31 out of 50"
python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icic.mp4" --input_refined_path="./output/${test_name}icic.mp4" --input_control_path="./input/${video_name}_preprocessed.mp4" --control_mask="./output/${test_name}_mask" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=30 --forward_steps=1
python src/detailrecover.py --input_source_path="./input/${video_name}_preprocessed" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0
for i in {31..49}
do
    echo "Processing step $((i+1)) out of 50"
    # step
    python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./input/${video_name}_preprocessed.mp4" --control_mask="./output/${test_name}_mask" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=$((i)) --forward_steps=1 --use_existing_noise
    # detail recovery
    python src/detailrecover.py --input_source_path="./input/${video_name}_preprocessed" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0
done

python src/getmask.py --input_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_mask"
python src/ProPainter/inference_propainter.py --video="./output/${test_name}icreflow.mp4" --mask="./output/${test_name}icreflow_mask" --output="./output/${test_name}icreflow_inpaint" --height=480 --width=720 --fp16 --save_frames --mask_dilation=2
python src/detailrecover.py --input_source_path="./input/${video_name}_preprocessed" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}_output.mp4" --blur_sigma=1.0 --inpaint_path="./output/${test_name}icreflow_inpaint/${test_name}icreflow/frames"

python src/IC-Light/video_edit_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./output/${test_name}_output.mp4" --output_path="./output/${test_name}ic.mp4" --prompt="${prompt}" --mix_bg --strength=0.4
python src/IC-Light/video_edit_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icic.mp4" --prompt="${prompt}" --mix_bg --strength=0.3

# renoise
echo "Processing step 41 out of 50"
python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icic.mp4" --input_refined_path="./output/${test_name}icic.mp4" --input_control_path="./input/${video_name}_preprocessed.mp4" --control_mask="./output/${test_name}_mask" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=40 --forward_steps=1
python src/detailrecover.py --input_source_path="./input/${video_name}_preprocessed" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=1.0
for i in {41..49}
do
    echo "Processing step $((i+1)) out of 50"
    # step
    python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./input/${video_name}_preprocessed.mp4" --control_mask="./output/${test_name}_mask" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=$((i)) --forward_steps=1 --use_existing_noise
    # detail recovery
    python src/detailrecover.py --input_source_path="./input/${video_name}_preprocessed" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=1.0
done

python src/getmask.py --input_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_mask"
python src/ProPainter/inference_propainter.py --video="./output/${test_name}icreflow.mp4" --mask="./output/${test_name}icreflow_mask" --output="./output/${test_name}icreflow_inpaint" --height=480 --width=720 --fp16 --save_frames --mask_dilation=1
python src/detailrecover.py --input_source_path="./input/${video_name}_preprocessed" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}_output.mp4" --blur_sigma=1.0 --inpaint_path="./output/${test_name}icreflow_inpaint/${test_name}icreflow/frames"