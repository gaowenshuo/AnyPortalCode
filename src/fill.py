import torch
import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from PIL import Image
import numpy as np

def dilate_mask(mask_tensor, kernel_size=5):
    """使用最大池化实现mask膨胀"""
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask_tensor.device)
    padding = kernel_size // 2
    
    dilated = F.conv2d(mask_tensor.float(), kernel, padding=padding) > 0
    return dilated.squeeze()

def preprocess_mask(mask_image, dilation_kernel=5):
    """预处理mask图像：转换为张量+膨胀处理"""
    mask_np = np.array(mask_image.convert("L"))
    mask_tensor = torch.from_numpy(mask_np).to('cuda') / 255.0
    mask_tensor = (mask_tensor > 0.5).float()
    
    if dilation_kernel > 0:
        mask_tensor = dilate_mask(mask_tensor, kernel_size=dilation_kernel)
    
    mask_np = mask_tensor.cpu().numpy()
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    
    return Image.fromarray((mask_np * 255).astype(np.uint8))

def main():
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description='Image Inpainting with Stable Diffusion')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask_image', type=str, required=True, help='Path to mask image')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for inpainting')
    args = parser.parse_args()

    # 加载原始图像和mask
    input_image = Image.open(args.input_image)
    original_mask = Image.open(args.mask_image)

    # 记录原始尺寸
    original_size = input_image.size

    # 预处理mask
    mask = preprocess_mask(original_mask, dilation_kernel=10)

    # 初始化inpaint管道
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to('cuda')

    # 执行inpainting
    result = pipe(
        prompt=args.prompt,
        negative_prompt="Blurry, Messy, Bad Quality, Watermark.",
        image=input_image,
        mask_image=mask,
        generator=torch.Generator("cpu").manual_seed(0),
        num_inference_steps=50
    ).images[0]

    # 调整大小并保存
    result = result.resize(original_size, Image.LANCZOS)
    result.save(args.output)
    print(f"Inpainting completed! Output saved to {args.output}")

if __name__ == "__main__":
    main()