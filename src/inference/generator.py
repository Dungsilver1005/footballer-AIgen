"""Run image generation with a prepared Stable Diffusion pipeline."""

from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

from .config import GenerationConfig, PROJECT_ROOT


_DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "inference_outputs"


def generate(
    pipe: StableDiffusionPipeline, generation_config: GenerationConfig
) -> list[Image]:
    """Generate images using the supplied pipeline and request settings."""
    generator = torch.Generator(device=pipe.device).manual_seed(
        generation_config.seed
    )
    result = pipe(
        prompt=generation_config.prompt,
        negative_prompt=generation_config.negative_prompt,
        width=generation_config.width,
        height=generation_config.height,
        num_inference_steps=generation_config.num_inference_steps,
        guidance_scale=generation_config.guidance_scale,
        num_images_per_prompt=generation_config.num_images,
        generator=generator,
    )
    return result.images


def save_images(
    images: list[Image],
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    seed: int | None = None,
) -> list[Path]:
    """Save images as PNG files and return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed_part = f"_seed{seed}" if seed is not None else ""
    saved_paths: list[Path] = []

    for index, image in enumerate(images, start=1):
        stem = f"{timestamp}{seed_part}_{index:02d}"
        collision = 0
        while True:
            suffix = f"_{collision}" if collision else ""
            path = directory / f"{stem}{suffix}.png"
            try:
                with path.open("xb") as file:
                    image.save(file, format="PNG")
            except FileExistsError:
                collision += 1
                continue
            saved_paths.append(path)
            break

    return saved_paths


def generate_and_save(
    pipe: StableDiffusionPipeline,
    generation_config: GenerationConfig,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Generate images, save them, and return their paths."""
    images = generate(pipe, generation_config)
    return save_images(images, output_dir=output_dir, seed=generation_config.seed)
