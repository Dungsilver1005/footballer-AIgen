"""Orchestrate construction of the WD 1.5 Beta3 base inference pipeline."""

from diffusers import StableDiffusionPipeline

from .config import AppConfig
from .model_loader import load_base_model
from .model_validator import validate_model_config
from .scheduler import configure_scheduler


def build_pipeline(config: AppConfig) -> StableDiffusionPipeline:
    """Validate, load, and configure the base model for inference."""
    validate_model_config(config.model)
    pipe = load_base_model(config.model)
    return configure_scheduler(pipe, config.scheduler)
