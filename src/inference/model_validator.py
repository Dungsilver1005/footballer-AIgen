"""Validation helpers for the local WD 1.5 Beta3 model configuration."""

from pathlib import Path
from typing import Final

from .config import BaseModelConfig


_REQUIRED_DIFFUSERS_FILES: Final[tuple[Path, ...]] = (
    Path("model_index.json"),
    Path("scheduler/scheduler_config.json"),
    Path("text_encoder/config.json"),
    Path("tokenizer/tokenizer_config.json"),
    Path("tokenizer/special_tokens_map.json"),
    Path("tokenizer/vocab.json"),
    Path("tokenizer/merges.txt"),
    Path("unet/config.json"),
    Path("vae/config.json"),
)
_SUPPORTED_DEVICES: Final[frozenset[str]] = frozenset({"cuda", "cpu"})
_SUPPORTED_TORCH_DTYPES: Final[frozenset[str]] = frozenset(
    {"float16", "float32", "bfloat16"}
)


def validate_model_config(config: BaseModelConfig) -> None:
    """Raise a clear error when the base model configuration is invalid."""
    _validate_checkpoint(config.pretrained_model_path)
    _validate_diffusers_config(config.diffusers_config_dir)
    _validate_device(config.device)
    _validate_torch_dtype(config.torch_dtype)


def _validate_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Pretrained model checkpoint does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Pretrained model checkpoint must be a file: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(
            "Pretrained model checkpoint must use the .safetensors extension: "
            f"{path}"
        )


def _validate_diffusers_config(config_dir: Path) -> None:
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Diffusers config directory does not exist: {config_dir}"
        )
    if not config_dir.is_dir():
        raise ValueError(f"Diffusers config path must be a directory: {config_dir}")

    missing_files = [
        config_dir / relative_path
        for relative_path in _REQUIRED_DIFFUSERS_FILES
        if not (config_dir / relative_path).is_file()
    ]
    if missing_files:
        missing_list = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Diffusers config directory is missing required file(s): {missing_list}"
        )


def _validate_device(device: str) -> None:
    if device not in _SUPPORTED_DEVICES:
        supported = ", ".join(sorted(_SUPPORTED_DEVICES))
        raise ValueError(f"Unsupported device '{device}'. Supported devices: {supported}.")

    if device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "CUDA was requested, but PyTorch could not be imported to check CUDA "
                "availability."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available in this environment."
            )


def _validate_torch_dtype(torch_dtype: str) -> None:
    if torch_dtype not in _SUPPORTED_TORCH_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_TORCH_DTYPES))
        raise ValueError(
            f"Unsupported torch_dtype '{torch_dtype}'. Supported values: {supported}."
        )
