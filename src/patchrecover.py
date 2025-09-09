import torch
import numpy as np
import os
import argparse
from PIL import Image
from torchvision import transforms
import imageio

def parse_args():
    parser = argparse.ArgumentParser(description='Detail Recovery Script')
    parser.add_argument('--input_source_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--input_target_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    parser.add_argument('--return_full_frame', action='store_true', help='Return full frame instead of cropped face')
    parser.add_argument('--blur_sigma', type=float, default=5.0, help='Sigma value for Gaussian blur')
    parser.add_argument('--inpaint_path', type=str, help='Path to the inpaint file')
    parser.add_argument('--using_existing_mask', type=str) #
    return parser.parse_args()

args = parse_args()
input_source_path = args.input_source_path
input_target_path = args.input_target_path
input_inpaint_path = args.inpaint_path
inpaint = False
if input_inpaint_path:
    inpaint = True
output_path = args.output_path
from transformers import AutoModelForImageSegmentation
rmbg = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
rmbg.eval()
device = torch.device('cuda')
rmbg = rmbg.to(device=device, dtype=torch.float32)

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def run_rmbg(imgs):
    H, W, C = imgs[0].shape
    assert C == 3
    feeds = []
    for img in imgs:
        assert img.shape == (H, W, C)
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(img, int(64 * round(W // 64 / 2)), int(64 * round(H // 64 / 2)))
        feeds.append(feed)
    feedh = torch.from_numpy(np.stack(feeds, axis=0)).float() / 255.0
    #print(feedh.shape)
    feedh = feedh.movedim(-1, 1).to(device=device, dtype=torch.float32)
    #print(feedh.shape)
    with torch.no_grad():
        alpha = rmbg(feedh)[-1].sigmoid()
        #print(alpha.shape)
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    #print(alpha.shape)
    return alpha

def process(target, source, inpainting, mode, blur_sigma, blend_factor, mask_source=None, mask_target=None):
    B, H, W, C = target.shape
    target_tensor = target.to(device)
    source_tensor = source.to(device)
    inpainting_tensor = inpainting.to(device)
    kernel_size = int(6 * int(blur_sigma) + 1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))
    blurred_target = gaussian_blur(target_tensor)
    blurred_source = gaussian_blur(source_tensor)
    
    if mode == "add":
        tensor_out = (source_tensor - blurred_source) + blurred_target
    elif mode == "multiply":
        tensor_out = source_tensor * blurred_target
    elif mode == "screen":
        tensor_out = 1 - (1 - source_tensor) * (1 - blurred_target)
    elif mode == "overlay":
        tensor_out = torch.where(blurred_target < 0.5, 2 * source_tensor * blurred_target, 1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "soft_light":
        tensor_out = (1 - 2 * blurred_target) * source_tensor**2 + 2 * blurred_target * source_tensor
    elif mode == "hard_light":
        tensor_out = torch.where(source_tensor < 0.5, 2 * source_tensor * blurred_target, 1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "difference":
        tensor_out = torch.abs(blurred_target - source_tensor)
    elif mode == "exclusion":
        tensor_out = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
    elif mode == "color_dodge":
        tensor_out = blurred_target / (1 - source_tensor)
    elif mode == "color_burn":
        tensor_out = 1 - (1 - blurred_target) / source_tensor
    elif mode == "divide":
        tensor_out = (source_tensor / blurred_source) * blurred_target
    else:
        tensor_out = source_tensor
    
    tensor_out = torch.lerp(target_tensor, tensor_out, blend_factor)

    tensor_out = source_tensor  # this is wrong

    if not args.return_full_frame:
        mask_target = mask_target.to(device)
        mask_source = mask_source.to(device)
        tensor_out = inpainting_tensor * (1 - mask_source) + tensor_out * mask_source
    tensor_out = torch.clamp(tensor_out, 0, 1)
    tensor_out = tensor_out.permute(0, 2, 3, 1).cpu().float()
    return (tensor_out,)

frames = []

source_frames = []
target_frames = []
for i in range(49):
    input_source = Image.open(os.path.join(input_source_path, f'frame_{i:03d}.png'))
    input_target = Image.open(os.path.join(input_target_path, f'frame_{i:03d}.png'))
    source_frames.append(np.array(input_source))
    target_frames.append(np.array(input_target))
frame_count = len(source_frames)
assert frame_count == len(target_frames)
batch_size = 49
mask_source_frames = []
mask_target_frames = []
using_existing_mask = args.using_existing_mask #
if using_existing_mask: #
    for i in range(49):
        mask_image = np.array(Image.open(os.path.join(using_existing_mask, f'{i:05d}.png')))
        mask_image = mask_image.astype(np.float32)
        mask_image = mask_image[..., np.newaxis] / 255.0
        mask_source_frames.append(mask_image)
        mask_target_frames.append(mask_image)
    mask_source_frames = np.array(mask_source_frames)
    mask_target_frames = np.array(mask_target_frames)
else: #
    for i in range(0, frame_count, batch_size):
        endpoint = min(i + batch_size, frame_count + 1)
        batch_source = source_frames[i:endpoint]
        batch_target = target_frames[i:endpoint]
        mask_source = run_rmbg(batch_source)
        mask_target = run_rmbg(batch_target)
        mask_source_frames.extend(mask_source)
        mask_target_frames.extend(mask_target)

for i in range(49):
    print(f"Processing frame {i}")
    input_source = Image.open(os.path.join(input_source_path, f'frame_{i:03d}.png'))
    input_target = Image.open(os.path.join(input_target_path, f'frame_{i:03d}.png'))
    mask_source = mask_source_frames[i]
    mask_target = mask_target_frames[i]
    mask_target_inpainting = torch.from_numpy(np.array(mask_target)).permute(2, 0, 1).permute(1, 2, 0).unsqueeze(0).float()
    mask_target_inpainting = mask_target_inpainting.squeeze(0).numpy() * 255
    mask_target_inpainting = mask_target_inpainting.astype(np.uint8)
    source_tensor = torch.from_numpy(np.array(input_source)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    target_tensor = torch.from_numpy(np.array(input_target)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mask_source = torch.from_numpy(np.array(mask_source)).permute(2, 0, 1).unsqueeze(0).float()
    mask_target = torch.from_numpy(np.array(mask_target)).permute(2, 0, 1).unsqueeze(0).float()
    import cv2
    if inpaint:
        inpainting = Image.open(os.path.join(input_inpaint_path, f'{i:04d}.png'))
        inpainting = np.array(inpainting)
    else:
        inpainting = np.array(input_target)
    inpainting_tensor = torch.from_numpy(inpainting).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    result = process(target_tensor, source_tensor, inpainting_tensor, 'add', args.blur_sigma, 1.0, mask_source=mask_source, mask_target=mask_target)[0]
    result = (result * 255).clip(0, 255).squeeze(0).numpy().astype(np.uint8)
    frames.append(result)

fps = 8
output_dir = os.path.splitext(output_path)[0]

# Create directory for frames if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save video and frames
with imageio.get_writer(output_path, fps=fps) as video_writer:
    for i, frame in enumerate(frames):
        video_writer.append_data(frame)
        # Save each frame as an image
        frame_image = Image.fromarray(frame)
        frame_image.save(os.path.join(output_dir, f"frame_{i:03d}.png"))


    

