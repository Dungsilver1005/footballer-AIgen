"""Load the local WD 1.5 Beta3 base Stable Diffusion pipeline."""

import hashlib

import torch
from diffusers import StableDiffusionPipeline

from .config import BaseModelConfig, BaseModelProvenance
from .model_validator import validate_model_config


_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def load_base_model(config: BaseModelConfig) -> StableDiffusionPipeline:
    """Load the local base and retain its source fingerprint for validation."""
    validate_model_config(config)
    checkpoint_path = config.pretrained_model_path.expanduser().resolve()
    before = checkpoint_path.stat()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)

    pipeline = StableDiffusionPipeline.from_single_file(
        str(checkpoint_path),
        config=str(config.diffusers_config_dir),
        local_files_only=True,
        torch_dtype=_TORCH_DTYPES[config.torch_dtype],
    )
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_ino,
    ):
        raise RuntimeError(f"Base checkpoint changed while loading: {checkpoint_path}")

    pipeline = pipeline.to(config.device)
    pipeline._project_base_model_info = BaseModelProvenance(
        checkpoint_path=checkpoint_path,
        sha256=digest.hexdigest(),
        unet=pipeline.unet,
        text_encoder=pipeline.text_encoder,
    )
    return pipeline
