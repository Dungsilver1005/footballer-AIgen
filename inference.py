"""Run local Stable Diffusion 1.5 inference with a Kohya-trained LoRA."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "stable_diffusion" / "v1-5-pruned-emaonly.safetensors"
LORA_DIRECTORY = PROJECT_ROOT / "outputs" / "lora"
OUTPUT_DIRECTORY = PROJECT_ROOT / "inference_outputs"


def parse_output_dim(value: str) -> Tuple[int, int]:
    """Parse a WIDTHxHEIGHT value accepted by Stable Diffusion."""
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Both output dimensions must be positive multiples of 8."
        ) from error

    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise argparse.ArgumentTypeError("Both output dimensions must be positive multiples of 8.")
    return width, height


def find_lora() -> Path:
    """Return the most recently modified LoRA checkpoint in the configured directory."""
    if not LORA_DIRECTORY.is_dir():
        raise FileNotFoundError(f"LoRA directory was not found: {LORA_DIRECTORY}")

    candidates = [
        path for path in LORA_DIRECTORY.glob("*.safetensors")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No .safetensors LoRA checkpoint was found in {LORA_DIRECTORY}. "
            "Train or copy a Kohya LoRA checkpoint there."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def enable_memory_optimizations(pipe, torch_module, cuda_available: bool) -> str:
    """Enable memory optimizations suitable for low-VRAM inference."""

    # Reduce VAE decoding memory usage.
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    else:
        pipe.vae.enable_slicing()

    if not cuda_available:
        return "CPU mode (CUDA is unavailable); this will be slow."

    # PyTorch 2.x / Diffusers uses SDPA by default.
    # xFormers is optional and is intentionally not used here.
    attention_mode = "PyTorch default/SDPA attention"

    # Prefer Accelerate CPU offload for low-VRAM GPUs.
    try:
        import accelerate  # noqa: F401

        pipe.enable_model_cpu_offload()

        return f"model CPU offload, VAE slicing, and {attention_mode}"

    except (ImportError, AttributeError) as error:
        # Fallback when Accelerate is unavailable.
        pipe.to("cuda")

        return (
            f"VAE slicing and {attention_mode}; "
            f"CPU offload unavailable ({error})"
        )


def load_pipeline(torch_module):
    """Load the local SD 1.5 checkpoint without converting it."""
    from diffusers import StableDiffusionPipeline

    cuda_available = torch_module.cuda.is_available()
    dtype = torch_module.float16 if cuda_available else torch_module.float32
    try:
        pipe = StableDiffusionPipeline.from_single_file(
            str(BASE_MODEL_PATH),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
    except TypeError:
        # Compatibility with older Diffusers releases that do not accept
        # local_files_only for from_single_file.
        pipe = StableDiffusionPipeline.from_single_file(
            str(BASE_MODEL_PATH),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    return pipe, cuda_available


def load_lora_weights_compatibly(pipe, lora_path: Path) -> None:
    """Load Kohya LoRA weights across current Diffusers/Transformers naming changes."""
    state_dict, network_alphas = pipe.lora_state_dict(
        str(lora_path.parent),
        weight_name=lora_path.name,
        local_files_only=True,
    )

    has_text_model_prefix = any(
        name.startswith("text_model.") for name, _ in pipe.text_encoder.named_modules()
    )
    has_serialized_text_model_prefix = any(
        key.startswith("text_encoder.text_model.") for key in state_dict
    )
    if not has_text_model_prefix and has_serialized_text_model_prefix:
        state_dict = {
            (key.replace("text_encoder.text_model.", "text_encoder.", 1)
             if key.startswith("text_encoder.text_model.")
             else key): value
            for key, value in state_dict.items()
        }
        network_alphas = {
            (key.replace("text_encoder.text_model.", "text_encoder.", 1)
             if key.startswith("text_encoder.text_model.")
             else key): value
            for key, value in (network_alphas or {}).items()
        }

    pipe.load_lora_into_unet(
        state_dict,
        network_alphas=network_alphas,
        unet=pipe.unet,
        _pipeline=pipe,
    )
    pipe.load_lora_into_text_encoder(
        state_dict,
        network_alphas=network_alphas,
        text_encoder=pipe.text_encoder,
        prefix=pipe.text_encoder_name,
        _pipeline=pipe,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="a professional football player, detailed portrait")
    parser.add_argument("--negative_prompt", default="low quality, blurry, deformed, text, watermark")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dim", type=parse_output_dim, default=(512, 512))
    return parser


def main() -> int:
    raw_args = sys.argv[1:]
    for index, argument in enumerate(raw_args[:-1]):
        if argument == "--output_dim" and raw_args[index + 1].startswith("-"):
            raw_args[index] = f"--output_dim={raw_args[index + 1]}"
            del raw_args[index + 1]
            break
    args = build_parser().parse_args(raw_args)
    if args.steps <= 0:
        print("Error: --steps must be greater than zero.", file=sys.stderr)
        return 2
    if args.cfg_scale < 0:
        print("Error: --cfg_scale must be zero or greater.", file=sys.stderr)
        return 2

    if not BASE_MODEL_PATH.is_file():
        print(
            f"Error: base model was not found at {BASE_MODEL_PATH}. "
            "Place v1-5-pruned-emaonly.safetensors at that path.",
            file=sys.stderr,
        )
        return 1

    try:
        import torch
    except Exception as error:
        print(
            f"Error: PyTorch could not be imported ({error}). "
            "Run this script with the project environment that has PyTorch and Diffusers installed.",
            file=sys.stderr,
        )
        return 1

    try:
        lora_path = find_lora()
        pipe, cuda_available = load_pipeline(torch)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(
            f"Error loading the SD 1.5 pipeline: {error}\n"
            "Check that your Diffusers/PyTorch versions support from_single_file and that "
            "the local model/config files are available and compatible.",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not cuda_available:
        print("Warning: CUDA is unavailable. Falling back to CPU; generation may take a long time.", file=sys.stderr)

    try:
        load_lora_weights_compatibly(pipe, lora_path)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(
            f"Error: could not load LoRA '{lora_path.name}': {error}\n"
            "Ensure it is a Diffusers-compatible Kohya SD 1.5 LoRA, not a full checkpoint "
            "or a LoRA trained for a different base architecture.",
            file=sys.stderr,
        )
        return 1

    try:
        optimization_message = enable_memory_optimizations(pipe, torch, cuda_available)
        generator_device = "cuda" if cuda_available else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        width, height = args.output_dim
        print(f"Using LoRA: {lora_path.name}")
        print(f"Memory strategy: {optimization_message}")
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            width=width,
            height=height,
            generator=generator,
            cross_attention_kwargs={"scale": args.lora_scale},
        ).images[0]
    except RuntimeError as error:
        message = str(error)
        if "out of memory" in message.lower() or "cuda" in message.lower():
            print(
                f"Error during generation: {message}\n"
                "This may be a VRAM problem. Keep 512x512, reduce --steps, close other GPU "
                "programs, and ensure Accelerate is installed so model CPU offload can be used.",
                file=sys.stderr,
            )
        else:
            print(f"Error during generation: {message}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as error:
        print(f"Error during generation: {error}", file=sys.stderr)
        return 1

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = OUTPUT_DIRECTORY / f"sd15_lora_seed{args.seed}_{timestamp}.png"
    try:
        image.save(output_path)
    except OSError as error:
        print(f"Error saving image to {output_path}: {error}", file=sys.stderr)
        return 1

    print(f"Saved image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())