"""Typed configuration for the local WD 1.5 Beta3 inference pipeline.

This module only describes settings. It does not import PyTorch, load model
weights, instantiate a scheduler, or validate optional adapter paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Optional


# config.py -> inference -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SHA-256 published for the checkpoint used by create_default_config():
# https://huggingface.co/waifu-diffusion/wd-1-5-beta3/blob/main/wd-beta3-base-fp16.safetensors
WD_BETA3_SHA256: Final[str] = (
    "d38e7795464024781aef8d28ef2101e9ddad3d6efc3fb09ca53b56f0d74e412c"
)


@dataclass(frozen=True)
class BaseModelProvenance:
    """Loader-computed source fingerprint, bound to the loaded components.

    This is trusted in-process provenance, not a certificate for arbitrary
    pipelines or for weights manually modified after loading.
    """

    checkpoint_path: Path
    sha256: str
    unet: object = field(repr=False, compare=False)
    text_encoder: object = field(repr=False, compare=False)


@dataclass
class BaseModelConfig:
    name: str
    pretrained_model_path: Path
    diffusers_config_dir: Path
    device: str = "cuda"
    torch_dtype: str = "float16"
    local_files_only: bool = True
    model_family: str = "sd2"
    prediction_type: str = "v_prediction"


@dataclass
class SchedulerConfig:
    name: str = "euler_a"


@dataclass
class LoRAConfig:
    enabled: bool = False
    path: Optional[Path] = None
    scale: float = 0.8


@dataclass
class IPAdapterConfig:
    enabled: bool = False
    model_path: Optional[Path] = None
    image_encoder_path: Optional[Path] = None
    reference_image_path: Optional[Path] = None
    scale: float = 1.0


@dataclass
class GenerationConfig:
    prompt: str = (
        "vnfootballer, anime style, Vietnamese male footballer, full body, "
        "entire body visible, head to toe, athletic build, short messy black hair, "
        "wearing a red football kit, standing on a stadium field, focused expression, dynamic pose"
    )
    negative_prompt: str = (
        "low quality, blurry, deformed, text, watermark, cropped, out of frame, "
        "cut off feet, cut off head, close-up, portrait, upper body only"
    )
    width: int = 768
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 42
    num_images: int = 1


@dataclass
class RuntimeConfig:
    enable_attention_slicing: bool = False
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_xformers: bool = False
    deterministic: bool = False


@dataclass
class OutputConfig:
    output_dir: Path
    file_prefix: str = "wd_beta3_base"
    save_metadata: bool = False


@dataclass
class AppConfig:
    project_root: Path
    model: BaseModelConfig
    output: OutputConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    ip_adapter: IPAdapterConfig = field(default_factory=IPAdapterConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def create_default_config(project_root: Optional[Path] = None) -> AppConfig:
    """Create the local WD 1.5 Beta3 baseline, resolving paths from the root."""
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else PROJECT_ROOT
    )

    return AppConfig(
        project_root=root,
        model=BaseModelConfig(
            name="WD 1.5 Beta3",
            pretrained_model_path=(
                root
                / "models"
                / "stable_diffusion"
                / "wd-1-5-beta3"
                / "wd-beta3-base-fp16.safetensors"
            ).resolve(),
            diffusers_config_dir=(
                root / "models" / "stable_diffusion" / "wd-1-5-beta3" / "config"
            ).resolve(),
        ),
        output=OutputConfig(output_dir=(root / "inference_outputs").resolve()),
    )
