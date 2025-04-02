import torch
import numpy as np
import os
import argparse
from PIL import Image
from torchvision import transforms
import imageio

def parse_args():
    parser = argparse.ArgumentParser(description='Detail Recovery Script')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    return parser.parse_args()

args = parse_args()
input_path = args.input_path
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
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


frames = []

input_frames = []
for i in range(49):
    input_source = Image.open(os.path.join(input_path, f'frame_{i:03d}.png'))
    input_frames.append(np.array(input_source))
frame_count = len(input_frames)

batch_size = 49
mask_frames = []
for i in range(0, frame_count, batch_size):
    endpoint = min(i + batch_size, frame_count + 1)
    batch_frames = input_frames[i:endpoint]
    mask = run_rmbg(batch_frames)
    mask_frames.extend(mask)

for i in range(49):
    mask = mask_frames[i]
    mask = np.repeat(mask, 3, axis=2)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(os.path.join(output_path, f"mask_{i:03d}.png"))
    

