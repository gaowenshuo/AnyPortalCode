export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2

python demo.py \
    --prompt "A black woman with curly hair in a red suit and sweater holding a computer is smiling in a cyberpunk city street with neon lights. The background is clear and bright." \
    --checkpoint_path "/home/gaows/TextDESIGN/DiffusionAsShader/transformer" \
    --output_dir "./output_woman6" \
    --input_path "/home/gaows/TextDESIGN/relightver20/input/woman6_preprocessed.mp4" \
    --repaint "/home/gaows/TextDESIGN/relightver20/output/gracewoman6icic/frame_000.png" \
    --gpu 2 \