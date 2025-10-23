#!/bin/bash
# conda activate designcog

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2



video_name="man9"
test_name="testman9-2"
prompt="A man with a hat is talking near a campfire in a dark outside."
prompt2="A campfire in a night is burning in a dark outside."

# preprocess the input video
python src/preprocess.py --input_path="./input/${video_name}.mp4"

# prepare optical flow
python src/getmask.py --input_path="./input/${video_name}_preprocessed" --output_path="./input/${video_name}_preprocessed_mask"
python src/MatAnyone/inference_matanyone.py -i="./input/${video_name}_preprocessed.mp4" -m="./input/${video_name}_preprocessed_mask/mask_000.png" -o="./input/${video_name}_preprocessed_mask_mat" -c="./src/MatAnyone/pretrained_models/matanyone.pth" --save_image
python src/ProPainter/inference_propainter.py --video="./input/${video_name}_preprocessed.mp4" --mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha" --output="./input/${video_name}_preprocessed_inpaint_mat" --height=480 --width=720 --fp16 --save_frames --mask_dilation=1

# prepare first frame
python src/IC-Light/video_gen_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./input/${video_name}_preprocessed.mp4" --output_path="./output/${test_name}icic.mp4" --prompt="${prompt}" --mix_bg --only_first_frame --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed_pha.mp4"

python src/fill.py --input_image="./output/${test_name}icic/frame_000.png" --mask_image="./input/${video_name}_preprocessed_mask/mask_000.png" --prompt="${prompt2}" --output="./output/${test_name}icic/frame_000_inpaint.png"

python ./src/DiffusionAsShader/demo.py \
    --prompt "${prompt2}" \
    --checkpoint_path "./src/DiffusionAsShader/transformer" \
    --output_dir "./src/DiffusionAsShader/output_${test_name}" \
    --input_path "./input/${video_name}_preprocessed_inpaint_mat/${video_name}_preprocessed/inpaint_out.mp4" \
    --repaint "./output/${test_name}icic/frame_000_inpaint.png" \
    --gpu 0 \

cp ./src/DiffusionAsShader/output_${test_name}/result.mp4 ./output/${test_name}bg.mp4

python src/IC-Light/video_relight_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./output/${test_name}bg.mp4" --output_path="./output/${test_name}icl.mp4" --prompt="${prompt}" --mix_bg --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed_pha.mp4"
python src/IC-Light/video_edit_new.py --input_fg_path="./input/${video_name}_preprocessed.mp4" --input_bg_path="./output/${test_name}icl.mp4" --output_path="./output/${test_name}ic.mp4" --prompt="${prompt}" --mix_bg --strength=0.65 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed_pha.mp4"

# renoise
echo "Processing step 16 out of 50"
python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}ic.mp4" --input_refined_path="./output/${test_name}ic.mp4" --input_control_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=15 --forward_steps=5
python src/patchrecover.py --input_source_path="./output/${test_name}ic" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha"
python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=20 --forward_steps=5 --use_existing_noise  --backward_steps=5
python src/patchrecover.py --input_source_path="./output/${test_name}ic" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha"
for i in {4..5}
do
    echo "Processing step $((5*i+1)) out of 50"
    python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=$((5*i)) --forward_steps=5 --use_existing_noise
    python src/patchrecover.py --input_source_path="./output/${test_name}ic" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha"
    python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=$((5*i+5)) --forward_steps=5 --use_existing_noise  --backward_steps=5
    python src/patchrecover.py --input_source_path="./output/${test_name}ic" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha"
done
for i in {15..19}
do
    echo "Processing step $((2*i+1)) out of 50"
    python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=$((2*i)) --forward_steps=2 --use_existing_noise
    python src/patchrecover.py --input_source_path="./output/${test_name}ic" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha"
    python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=$((2*i+2)) --forward_steps=2 --use_existing_noise  --backward_steps=2
    python src/patchrecover.py --input_source_path="./output/${test_name}ic" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha"
done
for i in {20..24}
do
    echo "Processing step $((2*i+1)) out of 50"
    python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=$((2*i)) --forward_steps=2 --use_existing_noise
    python src/detailrecover.py --input_source_path="./output/${test_name}ic" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha"
    python src/cogvideox-controlnet/controlnet_edit.py --input_path="./output/${test_name}icreflow.mp4" --input_refined_path="./output/${test_name}icreflow_refined.mp4" --input_control_path="./output/${test_name}ic.mp4" --output_path="./output/${test_name}icreflow.mp4" --prompt="${prompt}" --steps=50 --warmup_steps=$((2*i+2)) --forward_steps=1 --use_existing_noise  --backward_steps=1
    python src/detailrecover.py --input_source_path="./output/${test_name}ic" --input_target_path="./output/${test_name}icreflow" --output_path="./output/${test_name}icreflow_refined.mp4" --blur_sigma=2.0 --using_existing_mask="./input/${video_name}_preprocessed_mask_mat/${video_name}_preprocessed/pha"
done
