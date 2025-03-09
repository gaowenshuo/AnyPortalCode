import os
import torch
import numpy as np
import argparse
from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline, CogVideoXTransformer3DModel, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler, AutoencoderKLCogVideoX
from diffusers.utils import load_video, export_to_video
from PIL import Image
import imageio
from torchvision import transforms

# Argument parser
parser = argparse.ArgumentParser(description="Process video with CogVideoX")
parser.add_argument('--input_path', type=str, required=True, help='Path to the input video file')
parser.add_argument('--input_refined_path', type=str, required=True, help='Path to the refined input video file')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output video file')
parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
parser.add_argument('--steps', type=int, required=True, help='Number of steps to run the model')
parser.add_argument('--warmup_steps', type=int, required=True, help='Number of warmup steps')
parser.add_argument("--forward_steps", type=int, required=True, help="Number of forward steps")
parser.add_argument('--use_existing_noise', action='store_true', help='Use existing noise')
args = parser.parse_args()
# Set global seed
torch.manual_seed(args.warmup_steps)
np.random.seed(int(args.warmup_steps))



pretrained_model_name_or_path = "THUDM/CogVideoX-5b"

tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer"
)

text_encoder = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16
)

transformer = CogVideoXTransformer3DModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.float16
)

vae = AutoencoderKLCogVideoX.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float16
)

scheduler = CogVideoXDDIMScheduler.from_pretrained(
    pretrained_model_name_or_path, subfolder="scheduler"
)


pipe = CogVideoXPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
)

pipe.to('cuda')

pipe.enable_model_cpu_offload()
#pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

num_frames = 49
video_path = args.input_path
video_frames = load_video(video_path)[:num_frames]
refined_video_path = args.input_refined_path
refined_video_frames = load_video(refined_video_path)[:num_frames]
output_path = args.output_path
fps = 8
output_dir = os.path.splitext(output_path)[0]
# Create directory for frames if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device("cuda")
video_reader = imageio.get_reader(video_path, "ffmpeg")
frames = [transforms.ToTensor()(frame) for frame in video_reader]
refined_frames = [transforms.ToTensor()(frame) for frame in refined_video_frames]

video_reader.close()

resized_frames = []
resized_refined_frames = []
resize_transform = transforms.Resize((480, 720))

for frame in frames:
    resized_frame = resize_transform(frame)
    resized_frames.append(resized_frame * 2 - 1)
for frame in refined_frames:
    resized_frame = resize_transform(frame)
    resized_refined_frames.append(resized_frame * 2 - 1)

frames = resized_frames[:49]
refined_frames = resized_refined_frames[:49]

frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(torch.float16)
refined_frames_tensor = torch.stack(refined_frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(torch.float16)

print(frames_tensor.shape)
print("Encoding") # [1, 3, 49, 480, 720]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    with torch.no_grad():
        distribution = vae.encode(frames_tensor)[0]
        encoded_frames = distribution.mean # sample(generator=torch.Generator().manual_seed(args.seed))
        encoded_frames_var = distribution.std
print("Finished Encoding") # [1, 16, 13, 60, 90]
print(encoded_frames.shape)
latent_init = encoded_frames.permute(0, 2, 1, 3, 4)  
latent_init_var = encoded_frames_var.permute(0, 2, 1, 3, 4)
latent_z = vae.config.scaling_factor * latent_init
latent_z_var = vae.config.scaling_factor * latent_init_var

print(refined_frames_tensor.shape)
print("Encoding refined") # [1, 3, 49, 480, 720]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    with torch.no_grad():
        distribution = vae.encode(refined_frames_tensor)[0]
        encoded_refined_frames = distribution.mean # .sample(generator=torch.Generator().manual_seed(args.seed))
        encoded_refined_frames_var = distribution.std
print("Finished Encoding refined") # [1, 16, 13, 60, 90]
print(encoded_refined_frames.shape)
latent_init_refined = encoded_refined_frames.permute(0, 2, 1, 3, 4)  
latent_init_refined_var = encoded_refined_frames_var.permute(0, 2, 1, 3, 4)
latent_z_refined = vae.config.scaling_factor * latent_init_refined
latent_z_refined_var = vae.config.scaling_factor * latent_init_refined_var

# torch.save(latent_z, f"latent_{args.warmup_steps}.pt")
# torch.save(latent_z_refined, f"latent_refined_{args.warmup_steps}.pt")


eps = torch.randn_like(latent_z) * pipe.scheduler.init_noise_sigma

steps = args.steps
pipe.scheduler.set_timesteps(steps)
timesteps = pipe.scheduler.timesteps
print(timesteps)
warmup = args.warmup_steps
count = args.forward_steps
if args.use_existing_noise:
    forward_xt = torch.load(os.path.join(output_dir, "forward_xt.pt"), weights_only=True)
    forward_pred_original_sample = torch.load(os.path.join(output_dir, "forward_pred_original_sample.pt"), weights_only=True)
    latents_center = (forward_pred_original_sample - latent_z) / latent_z_var
    latents_center = latents_center.clamp(-100, 100)
    new_pred_original_sample = latents_center * latent_z_refined_var + latent_z_refined
    # for ddim
    assert warmup > 0, "When using existing noise, warmup steps must be greater than 0"
    timestep = timesteps[warmup - 1]
    prev_timestep = timesteps[warmup]
    alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[prev_timestep]
    beta_prod_t = 1 - alpha_prod_t
    a_t = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5
    b_t = alpha_prod_t_prev**0.5 - alpha_prod_t**0.5 * a_t
    prev_sample = a_t * forward_xt + b_t * new_pred_original_sample
    latents = prev_sample
    # for ddim
else:
    latents = pipe.scheduler.add_noise(latent_z_refined, eps, timesteps[warmup])

prompt = args.prompt
text_input = pipe.tokenizer([prompt], padding="max_length",
                                    max_length=pipe.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
uncond_input = pipe.tokenizer(["Blurry, pixelated, overexposed, noisy, distorted shapes, inconsistent frames, unnatural motion, and cluttered background."], padding="max_length",
                                    max_length=text_input.input_ids.shape[-1],
                                    return_tensors="pt")
with torch.no_grad():
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
del pipe.text_encoder
torch.cuda.empty_cache()

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
prompt_embeds = text_embeddings

image_rotary_emb = (
            pipe._prepare_rotary_positional_embeddings(480, 720, latents.size(1), device)
            if pipe.transformer.config.use_rotary_positional_embeddings
            else None
        )

forward_latent = None
forward_xt = None
for i, t in enumerate(timesteps):
    latents = latents.to(prompt_embeds.dtype)
    if i < warmup:
        continue
    print("noise step", t)
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    timestep = t.expand(latent_model_input.shape[0])
    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + 6 * (noise_pred_text - noise_pred_uncond)
    with torch.no_grad():
        forward_xt = latents.clone()
    latents, pred_original_sample = pipe.scheduler.step(
                                            noise_pred,
                                            t,
                                            latents,
                                            return_dict=False,
                                        )
    latents = latents.to(prompt_embeds.dtype)
    with torch.no_grad():
        forward_latent = latents.clone()
        forward_pred_original_sample = pred_original_sample.clone() 
    count = count - 1
    if count == 0:
        latents = pred_original_sample.to(prompt_embeds.dtype)
        break


with torch.no_grad():
    torch.save(forward_xt, os.path.join(output_dir, "forward_xt.pt"))
    torch.save(forward_pred_original_sample, os.path.join(output_dir, "forward_pred_original_sample.pt"))

latents = latents.permute(0, 2, 1, 3, 4)
latents = 1 / vae.config.scaling_factor * latents

del pipe.transformer
torch.cuda.empty_cache()

print("Decoding") # [1, 16, 13, 60, 90]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    with torch.no_grad():
        decoded_frames = vae.decode(latents).sample
print("Finished Decoding") # [1, 3, 49, 480, 720]

frames = decoded_frames[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
frames = np.clip((frames + 1) * 127, 0, 255)
frames = frames.astype(np.uint8)

# Save video and frames
with imageio.get_writer(output_path, fps=fps) as video_writer:
    for i, frame in enumerate(frames):
        video_writer.append_data(frame)
        # Save each frame as an image
        frame_image = Image.fromarray(frame)
        frame_image.save(os.path.join(output_dir, f"frame_{i:03d}.png"))