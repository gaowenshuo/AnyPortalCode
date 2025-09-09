import imageio
import argparse
import numpy as np
import torch
import os

def process_video(input_path, target_width=720, target_height=480):
    # Generate output paths
    base, ext = os.path.splitext(input_path)
    output_video_path = f"{base}_preprocessed{ext}"
    output_frames_dir = f"{base}_preprocessed"
    os.makedirs(output_frames_dir, exist_ok=True)

    # Open the input video
    reader = imageio.get_reader(input_path, 'ffmpeg')
    fps = reader.get_meta_data()['fps']
    frame_count = reader.count_frames()
    width, height = reader.get_meta_data()['size']

    # Define the writer object
    writer = imageio.get_writer(output_video_path, fps=8)

    # Process the first 49 frames
    for i in range(49):
        j = i
        if i >= frame_count:
            j = i % frame_count
        frame = reader.get_data(j)

        # Center crop to 480x720
        center_x, center_y = width // 2, height // 2
        crop_width, crop_height = target_width, target_height
        width_factor = float(width) / float(crop_width)
        height_factor = float(height) / float(crop_height)
        width_factor = min(width_factor, height_factor)
        height_factor = min(width_factor, height_factor)
        crop_width = int(crop_width * width_factor)
        crop_height = int(crop_height * height_factor)
        x1 = max(0, center_x - crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        x2 = min(width, center_x + crop_width // 2)
        y2 = min(height, center_y + crop_height // 2)
        cropped_frame = frame[y1:y2, x1:x2]

        # Resize to 480x720
        import torchvision.transforms as transforms

        # Convert the frame to a PyTorch tensor
        frame_tensor = torch.from_numpy(cropped_frame).permute(2, 0, 1).float() / 255.0

        # Define the transform to resize the frame
        transform = transforms.Compose([
            transforms.Resize((target_height, target_width)),
        ])

        # Apply the transform
        resized_frame_tensor = transform(frame_tensor)

        # Convert the tensor back to a numpy array
        resized_frame = (resized_frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Write the frame to the output video
        writer.append_data(resized_frame)

        # Save the frame as an image
        frame_output_path = os.path.join(output_frames_dir, f"frame_{i:03d}.png")
        imageio.imwrite(frame_output_path, resized_frame)

    # Release everything if job is finished
    reader.close()
    writer.close()
    print("Video processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('--input_path', type=str, help='Path to the input video file')
    parser.add_argument('--width', type=int, default=720, help='Width of the output video')
    parser.add_argument('--height', type=int, default=480, help='Height of the output video')
    args = parser.parse_args()

    process_video(args.input_path, args.width, args.height)
