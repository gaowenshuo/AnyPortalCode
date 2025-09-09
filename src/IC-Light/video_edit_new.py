import os
import math
import numpy as np
import torch
import safetensors.torch as sf
import torch.nn.functional as F
import os
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoModelForImageSegmentation
from enum import Enum
from torch.hub import download_url_to_file
import imageio
import argparse

# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
rmbg.eval()

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = './src/IC-Light/models/iclight_sd15_fc.safetensors'

if not os.path.exists(model_path):
    print("Downloading model...")
    download_url_to_file(url='https://hf-mirror.com/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP
def rearrange_0(tensor, f):
    F, C, H, W = tensor.size()
    tensor = torch.permute(torch.reshape(tensor, (F // f, f, C, H, W)), (0, 2, 1, 3, 4))
    return tensor


def rearrange_1(tensor):
    B, C, F, H, W = tensor.size()
    return torch.reshape(torch.permute(tensor, (0, 2, 1, 3, 4)), (B * F, C, H, W))


def rearrange_3(tensor, f):
    F, D, C = tensor.size()
    return torch.reshape(tensor, (F // f, f, D, C))


def rearrange_4(tensor):
    B, F, D, C = tensor.size()
    return torch.reshape(tensor, (B * F, D, C))

class CrossFrameAttnProcessor2_0:
    """
    Cross frame attention processor with scaled_dot_product attention of Pytorch 2.0.

    Args:
        batch_size: The number that represents actual batch size, other than the frames.
            For example, calling unet with a single prompt and num_images_per_prompt=1, batch_size should be equal to
            2, due to classifier-free guidance.
    """

    def __init__(self, batch_size=2):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.batch_size = batch_size

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Cross Frame Attention
        if not is_cross_attention:
            video_length = max(1, key.size()[0] // self.batch_size)
            first_frame_index = [0] * video_length

            # rearrange keys to have batch and frames in the 1st and 2nd dims respectively
            key = rearrange_3(key, video_length)
            key = key[:, first_frame_index]
            # rearrange values to have batch and frames in the 1st and 2nd dims respectively
            value = rearrange_3(value, video_length)
            value = value[:, first_frame_index]

            # rearrange back to original shape
            key = rearrange_4(key)
            value = rearrange_4(value)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

unet.set_attn_processor(CrossFrameAttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    feed = resize_without_crop(img, 512, 512)
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        alpha = rmbg(feed)[-1].sigmoid()
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    
    image_width = (int(image_width) // 64) * 64
    image_height = (int(image_height) // 64) * 64
    
    bg_source = BGSource(bg_source)

    if bg_source == BGSource.UPLOAD:
        pass
    elif bg_source == BGSource.UPLOAD_FLIP:
        input_bg = np.fliplr(input_bg)
    elif bg_source == BGSource.GREY:
        input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(224, 32, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(32, 224, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(224, 32, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(32, 224, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong background source!'

    rng = torch.Generator(device=device).manual_seed(seed)

    
    fg = [resize_without_crop(f, image_width, image_height) for f in input_fg]
    bg = [resize_without_crop(b, image_width, image_height) for b in input_bg]
    concat_conds = numpy2pytorch([fg[0], fg[1]]).to(device=vae.device, dtype=vae.dtype)
    # print(concat_conds.shape)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    # print(concat_conds.shape)
    #concat_conds = concat_conds.view(concat_conds.shape[0] // 2, concat_conds.shape[1] * 2, concat_conds.shape[2], concat_conds.shape[3])
    # print(concat_conds.shape)

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    #latents = t2i_pipe(
    #    prompt_embeds=conds,
    #    negative_prompt_embeds=unconds,
    #    width=image_width,
    #    height=image_height,
    #    num_inference_steps=steps,
    #    num_images_per_prompt=num_samples,
    #    generator=rng,
    #    output_type='latent',
    #    guidance_scale=cfg,
    #    cross_attention_kwargs={'concat_conds': concat_conds},
    #).images.to(vae.dtype) / vae.config.scaling_factor

    #pixels = vae.decode(latents).sample
    #pixels = pytorch2numpy(pixels)
    # this line is to editing
    pixels = input_bg
    #

    image_width = image_width * 2
    image_height = image_height * 2
    image_width = (int(image_width) // 64) * 64
    image_height = (int(image_height) // 64) * 64

    pixels = [resize_without_crop(
        image=p,
        target_width=image_width,
        target_height=image_height)
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = [resize_without_crop(f, image_width, image_height) for f in input_fg]
    bg = [resize_without_crop(b, image_width, image_height) for b in input_bg]
    concat_conds = numpy2pytorch([fg[0], fg[1]]).to(device=vae.device, dtype=vae.dtype)
    # print(concat_conds.shape)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    # print(concat_conds.shape)
    #concat_conds = concat_conds.view(concat_conds.shape[0] // 2, concat_conds.shape[1] * 2, concat_conds.shape[2], concat_conds.shape[3])
    # print(concat_conds.shape)

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels, [fg, bg]


@torch.inference_mode()
def process_relight(input_fg, input_bg, input_mask, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source): #
    if input_mask[0] is not None: #
        matting0 = np.mean(input_mask[0].astype(np.float32), axis=-1)
        matting0 = matting0[..., np.newaxis] / 255.0
        input_fg0 = 127 + (input_fg[0].astype(np.float32) - 127) * matting0
        input_fg0 = input_fg0.clip(0, 255).astype(np.uint8)
        matting1 = np.mean(input_mask[1].astype(np.float32), axis=-1)
        matting1 = matting1[..., np.newaxis] / 255.0
        input_fg1 = 127 + (input_fg[1].astype(np.float32) - 127) * matting1
        input_fg1 = input_fg1.clip(0, 255).astype(np.uint8)
    else: #
        input_fg0, matting0 = run_rmbg(input_fg[0]) #
        input_fg1, matting1 = run_rmbg(input_fg[1]) #
    input_fg = [input_fg0, input_fg1]
    matting = [matting0, matting1]
    height, width, _ = input_fg[0].shape
    results, extra_images = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    # print(matting.shape)
    # print(results[0].shape)
    # print(input_bg.shape)
    results = [resize_without_crop(x, width, height) for x in results]
    # print(results[0].shape)
    # print(input_bg.shape)
    if not args.change_bg:
        results = [x * matting[ii] + input_bg[ii] * (1 - matting[ii]) for ii, x in enumerate(results)]
    results = [x.clip(0, 255).astype(np.uint8) for x in results]
    # print(results[0].shape)
    return results + extra_images


@torch.inference_mode()
def process_normal(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg, sigma=16)

    print('left ...')
    left = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.LEFT.value)[0][0]

    print('right ...')
    right = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.RIGHT.value)[0][0]

    print('bottom ...')
    bottom = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.BOTTOM.value)[0][0]

    print('top ...')
    top = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.TOP.value)[0][0]

    inner_results = [left * 2.0 - 1.0, right * 2.0 - 1.0, bottom * 2.0 - 1.0, top * 2.0 - 1.0]

    ambient = (left + right + bottom + top) / 4.0
    h, w, _ = ambient.shape
    matting = resize_without_crop((matting[..., 0] * 255.0).clip(0, 255).astype(np.uint8), w, h).astype(np.float32)[..., None] / 255.0

    def safa_divide(a, b):
        e = 1e-5
        return ((a + e) / (b + e)) - 1.0

    left = safa_divide(left, ambient)
    right = safa_divide(right, ambient)
    bottom = safa_divide(bottom, ambient)
    top = safa_divide(top, ambient)

    u = (right - left) * 0.5
    v = (top - bottom) * 0.5

    sigma = 10.0
    u = np.mean(u, axis=2)
    v = np.mean(v, axis=2)
    h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
    z = np.zeros_like(h)

    normal = np.stack([u, v, h], axis=2)
    normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
    normal = normal * matting + np.stack([z, z, 1 - z], axis=2) * (1 - matting)

    results = [normal, left, right, bottom, top] + inner_results
    results = [(x * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for x in results]
    return results


quick_prompts = [
    'beautiful woman',
    'handsome man',
    'beautiful woman, cinematic lighting',
    'handsome man, cinematic lighting',
    'beautiful woman, natural lighting',
    'handsome man, natural lighting',
    'beautiful woman, neo punk lighting, cyberpunk',
    'handsome man, neo punk lighting, cyberpunk',
]
quick_prompts = [[x] for x in quick_prompts]


class BGSource(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"

def main(input_fg_path, input_bg_path, output_path, prompt, image_width, image_height, using_existing_mask): #
    reader_fg = imageio.get_reader(input_fg_path)
    reader_bg = imageio.get_reader(input_bg_path)
    if using_existing_mask: #
        reader_mask = imageio.get_reader(using_existing_mask) #
    else: #
        reader_mask = None #
    base, ext = os.path.splitext(output_path)
    output_dir = base
    os.makedirs(output_dir, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=8)
    num_samples = 2
    seed = int(100000 * args.strength)
    steps = 20
    a_prompt = "best quality"
    n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    cfg = 7.0
    highres_scale = 1.0
    highres_denoise = args.strength
    bg_source = BGSource.UPLOAD.value

    first_frame_fg = reader_fg.get_data(0)
    first_frame_bg = reader_bg.get_data(0)
    if using_existing_mask: #
        first_frame_mask = reader_mask.get_data(0) #
    else: #
        first_frame_mask = None #

    for i in range(0, 49):
        print("Processing frame", i)
        fg_frame = reader_fg.get_data(i)
        bg_frame = reader_bg.get_data(i)
        fg_frame = resize_without_crop(fg_frame, image_width, image_height)
        bg_frame = resize_without_crop(bg_frame, image_width, image_height)
        first_fg_frame = resize_without_crop(first_frame_fg, image_width, image_height)
        first_bg_frame = resize_without_crop(first_frame_bg, image_width, image_height)
        if using_existing_mask: #
            mask_frame = reader_mask.get_data(i) #
            mask_frame = resize_without_crop(mask_frame, image_width, image_height) #
            first_mask_frame = resize_without_crop(first_frame_mask, image_width, image_height) #
        else: #
            mask_frame = None
            first_mask_frame = None
        
        frame_results = process_relight(
            input_fg=[first_fg_frame, fg_frame],
            input_bg=[first_bg_frame, bg_frame],
            input_mask=[first_mask_frame, mask_frame], #
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            bg_source=bg_source
        )
        if args.mix_bg:
            result0 = frame_results[0] * 1.0 + first_bg_frame * 0.0
            result0 = result0.clip(0, 255).astype(np.uint8)
            result1 = frame_results[1] * 1.0 + bg_frame * 0.0
            result1 = result1.clip(0, 255).astype(np.uint8)
        else:
            result0 = frame_results[0]
            result1 = frame_results[1]
        writer.append_data(result1)
        frame_output_path0 = os.path.join(output_dir, f"frame0_{i:03d}.png")
        frame_output_path1 = os.path.join(output_dir, f"frame_{i:03d}.png")
        imageio.imwrite(frame_output_path0, result0)
        imageio.imwrite(frame_output_path1, result1)


    reader_fg.close()
    reader_bg.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and create a result video.")
    parser.add_argument("--input_fg_path", type=str, required=True, help="Path to the foreground video.")
    parser.add_argument("--input_bg_path", type=str, required=True, help="Path to the background video.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output video file.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the video generation.")
    parser.add_argument("--mix_bg", action="store_true", help="Mix the background with the generated image.")
    parser.add_argument("--strength", type=float, default=0.5, help="Strength of the highres denoise.")
    parser.add_argument("--change_bg", action="store_true", help="Change the background.")
    parser.add_argument("--width", type=int, default=720, help="Width of the output video.")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video.")
    parser.add_argument("--using_existing_mask", type=str) #

    args = parser.parse_args()
    main(args.input_fg_path, args.input_bg_path, args.output_path, args.prompt, args.width, args.height, args.using_existing_mask) #
