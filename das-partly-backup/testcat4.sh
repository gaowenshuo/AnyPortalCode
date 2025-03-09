export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2

python demo.py \
    --prompt "A cat is standing in a broad living room on a wooden floor. The sunlight shine from the window and shine the cat's shadow and create a warm scene. The background is clear and bright." \
    --checkpoint_path "/home/gaows/TextDESIGN/DiffusionAsShader/transformer" \
    --output_dir "./output_cat4" \
    --input_path "/home/gaows/TextDESIGN/relightver20/input/cat4_preprocessed.mp4" \
    --repaint "/home/gaows/TextDESIGN/relightver20/output/gracecat4icic/frame_000.png" \
    --gpu 2 \