import torch
import numpy as np
import os
import argparse
from PIL import Image
import imageio
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='LAB Channel Transfer')
    parser.add_argument('--input_source_path', type=str, required=True, help='Path to source frames')
    parser.add_argument('--input_target_path', type=str, required=True, help='Path to target frames')
    parser.add_argument('--output_path', type=str, required=True, help='Output video path')
    return parser.parse_args()

def transfer_l_channel(source_frame, target_frame):
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source_frame, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target_frame, cv2.COLOR_RGB2LAB)
    
    # Replace target's L channel with source's L channel
    transferred_lab = np.stack([
        source_lab[:, :, 0],  # L channel from source
        target_lab[:, :, 1],   # A channel from target
        target_lab[:, :, 2]    # B channel from target
    ], axis=2)
    
    # Convert back to RGB
    return cv2.cvtColor(transferred_lab, cv2.COLOR_LAB2RGB)

def process_frames(source_path, target_path, output_path):
    frames = []
    
    # Process all 49 frames
    for i in range(49):
        # Read frames
        source_frame = np.array(Image.open(os.path.join(source_path, f'frame_{i:03d}.png')).convert('RGB'))
        target_frame = np.array(Image.open(os.path.join(target_path, f'frame_{i:03d}.png')).convert('RGB'))
        
        # Perform LAB channel transfer
        result = transfer_l_channel(source_frame, target_frame)
        
        # Convert back to uint8
        result = (result * 255).astype(np.uint8) if result.dtype == np.float32 else result
        frames.append(result)

    # Save video
    fps = 8
    with imageio.get_writer(output_path, fps=fps) as video_writer:
        for frame in frames:
            video_writer.append_data(frame)
    
    # Save frames
    output_dir = os.path.splitext(output_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(output_dir, f"frame_{i:03d}.png"))

if __name__ == "__main__":
    args = parse_args()
    process_frames(
        args.input_source_path,
        args.input_target_path,
        args.output_path
    )