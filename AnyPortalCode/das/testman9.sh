export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

python demo.py \
    --prompt "A man with glasses and a hat is talking on a beach near a sea with waves. The golden sun shines on the sea, creating a glistening and rippling effect. The background is clear and bright." \
    --checkpoint_path "/home/gaows/TextDESIGN/DiffusionAsShader/transformer" \
    --output_dir "./output_man9" \
    --input_path "/home/gaows/TextDESIGN/relightver20/input/man9_preprocessed.mp4" \
    --repaint "/home/gaows/TextDESIGN/relightver20/output/graceman9icic/frame_000.png" \
    --gpu 1 \