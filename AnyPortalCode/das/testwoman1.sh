export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

python demo.py \
    --prompt "A beautiful woman holding a glass is in a park outside with warm atmosphere." \
    --checkpoint_path "/home/gaows/TextDESIGN/DiffusionAsShader/transformer" \
    --output_dir "./output_woman1" \
    --input_path "/home/gaows/TextDESIGN/relightver20/input/woman1_preprocessed.mp4" \
    --repaint "/home/gaows/TextDESIGN/relightver20/output/gracewoman1icic/frame_000.png" \
    --gpu 1 \