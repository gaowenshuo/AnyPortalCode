import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import torch
import numpy as np
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler, AutoencoderKLCogVideoX
from diffusers.utils import load_video, export_to_video
from controlnet_aux import CannyDetector, HEDdetector

from controlnet_pipeline import ControlnetCogVideoXPipeline
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from cogvideo_controlnet import CogVideoXControlnet

pretrained_model_name_or_path = "THUDM/CogVideoX-2b"

tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer"
)

text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)

transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="transformer"
)

vae = AutoencoderKLCogVideoX.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)

scheduler = CogVideoXDDIMScheduler.from_pretrained(
    pretrained_model_name_or_path, subfolder="scheduler"
)

controlnet = CogVideoXControlnet.from_pretrained('TheDenk/cogvideox-2b-controlnet-canny-v1')

pipe = ControlnetCogVideoXPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    controlnet=controlnet,
    scheduler=scheduler,
)
pipe = pipe.to(dtype=torch.float16, device='cuda')

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

controlnet_processor = CannyDetector()

num_frames = 49
video_path = './resources/car.mp4'
video_frames = load_video(video_path)[:num_frames]
controlnet_frames = [controlnet_processor(x) for x in video_frames]

output = pipe(
    controlnet_frames=controlnet_frames,
    prompt='red car is moving on the ocean waves, beautiful waves',
    height=480,
    width=720,
    num_frames=49,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(42),
    controlnet_weights=0.8,
    controlnet_guidance_start=0.0,
    controlnet_guidance_end=0.8,
)

export_to_video(output.frames[0], 'out.mp4', fps=16)