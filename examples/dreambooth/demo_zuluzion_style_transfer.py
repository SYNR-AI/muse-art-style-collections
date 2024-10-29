import time
_TIC = time.perf_counter()

import argparse
import os
import torch
import random
import numpy as np

from diffusers import FluxPipeline, FluxImg2ImgPipeline
from zulusion.pipeline_flux_img2img_cfg import FluxImg2ImgCFGPipeline
from PIL import Image
print(f" -- import modules, time cost: {time.perf_counter() - _TIC}")

PRETRAINED_MODEL_PATH = "/mnt2/share/huggingface_models/FLUX.1-dev"


def set_all_seed(seed):
    # Setting seed for Python's built-in random module, NumPy, PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Ensuring reproducibility for convolutional layers in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(seed, device):
    generator = None
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(s) for s in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    return generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply style-transfer for character images via LoRA and prompts")

    parser.add_argument("--flux_model_path", type=str, default=PRETRAINED_MODEL_PATH,
                        help="FLUX.1-dev base model path.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to save output images, if not specified, results will be saved in the same folder of checkpoint")

    parser.add_argument("--trigger", type=str, default=None,
                        help="Trigger word of this LoRA model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA model path to load")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="LoRA scale, used in FluxPipeline..joint_attention_kwargs when denoising")

    parser.add_argument("--prompts", type=str, nargs='+', default=["",],
                        help="Prompts, if nothing provided, use default prompt-sentence(s).")
    parser.add_argument("--images", type=str, nargs='+', default=["",],
                        help="Images, shall have the same length as prompts.")

    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompts (currently only support one prompt for all), default is None.")
    parser.add_argument("--true_cfg", type=float, default=1.0,
                        help="True Classifier-Free Guidance scale, only useful when `true_cfg` > 1.0 and FluxCFGPipeline enabled")
    parser.add_argument("--step_start_cfg", type=int, default=5,
                        help="Step to start `true_cfg`. Notice: this should be less than `num_inference_steps`")

    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Guidance scale, used if transformer.config.guidance_embeds enabled")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for all builtin modules and generator")
    parser.add_argument("--height", type=int, default=1344,
                        help="Output size.height")
    parser.add_argument("--width", type=int, default=768,
                        help="Output size.width")
    parser.add_argument("--strength", type=float, default=1.0,
                        help=(
                            "Strength (`float`, *optional*, defaults to 1.0):"
                            "Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a"
                            "starting point and more noise is added the higher the `strength`. The number of denoising steps depends"
                            "on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising"
                            "process runs for the full number of iterations specified in `num_inference_steps`. A value of 1"
                            "essentially ignores `image`."))
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Sampling steps, use diffusers default=20")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="How many images would be sampled per prompt, default=1")

    parser.add_argument(
        "--jobs_idx",
        type=int,
        default=None,
        help="Split prompts, this job's index.")
    parser.add_argument(
        "--jobs_all",
        type=int,
        default=None,
        help="Split prompts, the total jobs number.")

    args = parser.parse_args()

    max_length = max(len(args.prompts), len(args.images))
    for _ in range(max_length - len(args.prompts)):
        args.prompts.append("")
    for _ in range(max_length - len(args.images)):
        args.images.append("")
    args.prompts_ids = list(range(max_length))
    if isinstance(args.jobs_idx, int) and isinstance(args.jobs_all, int):
        sep = int(args.jobs_idx)
        tot = int(args.jobs_all)
        if 0 <= sep and sep < tot:
            args.prompts = args.prompts[sep::tot]
            args.prompts_ids = args.prompts_ids[sep::tot]
            print(f" -- Split tasks: idx={sep}, [{sep+1}/{tot}]")

    args.enable_cfg_pipeline = args.negative_prompt is not None

    return args


def main(args):
    _TIC = time.perf_counter()
    if args.enable_cfg_pipeline:
        pipe = FluxImg2ImgCFGPipeline.from_pretrained(
            args.flux_model_path,
            torch_dtype=torch.bfloat16).to("cuda")
        print(" -- Switch to flux i2i+cfg pipeline")
    else:
        pipe = FluxImg2ImgPipeline.from_pretrained(
            args.flux_model_path,
            torch_dtype=torch.bfloat16).to("cuda")
        print(" -- Switch to flux i2i pipeline (w/o cfg)")
    print(f" -- init pipe, time cost: {time.perf_counter() - _TIC}")

    for i, prompt in enumerate(args.prompts):
        print(f"    {prompt}")
    num_test = len(args.prompts)

    # Enable LoRA (format from diffusers trained)
    _TIC = time.perf_counter()
    prompt_prefix_list =[]
    adapter_names = []
    adapter_weights = []

    if isinstance(args.trigger, str):
        args.trigger = args.trigger.strip()
        if args.trigger != "":
            prompt_prefix_list.append(args.trigger)
    if args.lora_path is not None:
        # Load character-lora:
        adapter_names.append("lora_1")
        adapter_weights.append(args.lora_scale)
        pipe.load_lora_weights(args.lora_path, adapter_name=adapter_names[0])

    print(f" -- Set LoRA(s) into pipeline: {time.perf_counter() - _TIC:.03f} sec")

    prompt_prefix = ", ".join(prompt_prefix_list)
    print(f" -- Prompt prefix: {prompt_prefix}")

    os.makedirs(args.out_dir, exist_ok=True)

    for i, prompt in zip(args.prompts_ids, args.prompts):
        set_all_seed(args.seed)
        generator = get_generator(args.seed, "cuda")

        if len(adapter_names) > 0:
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

        if prompt_prefix != "":
            input_prompt = f"{prompt_prefix}, {prompt}"
        else:
            input_prompt = prompt

        input_image = Image.open(args.images[i]).convert("RGB")
        if args.enable_cfg_pipeline:
            images = pipe(
                prompt=input_prompt,
                image=input_image,
                negative_prompt=args.negative_prompt,
                true_cfg=args.true_cfg,
                step_start_cfg=args.step_start_cfg,
                height=args.height, width=args.width,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                guidance_scale=args.guidance_scale,
                generator=generator).images
        else:
            images = pipe(
                prompt=input_prompt,
                image=input_image,
                height=args.height, width=args.width,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                guidance_scale=args.guidance_scale,
                generator=generator).images

        for j, img in enumerate(images):
            out_name_img = f"__prompt{i}_sample{j}.jpg"
            img.save(os.path.join(args.out_dir, out_name_img))



if __name__ == '__main__':
    main(args=parse_args())
