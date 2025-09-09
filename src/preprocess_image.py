import imageio
import argparse
import numpy as np
import torch
import os

def process_image(input_path):
    # Generate output paths
    base, ext = os.path.splitext(input_path)
    output_video_path = f"{base}_preprocessed.mp4"
    output_frames_dir = f"{base}_preprocessed"
    os.makedirs(output_frames_dir, exist_ok=True)

    # Open the input image
    image = imageio.imread(input_path)
    height, width, _ = image.shape

    # Define the writer object
    writer = imageio.get_writer(output_video_path, fps=8)

    # Process 49 frames
    for i in range(49):
        frame = image

        # Center crop to 480x720
        center_x, center_y = width // 2, height // 2
        crop_width, crop_height = 720, 480
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
            transforms.Resize((480, 720))
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
    writer.close()
    print("Video processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image into a video.')
    parser.add_argument('--input_path', type=str, help='Path to the input image file')
    args = parser.parse_args()

    process_image(args.input_path)
