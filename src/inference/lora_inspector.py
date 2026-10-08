"""Inspect LoRA checkpoints and the pre-load state of an existing pipeline.

Only alpha values are materialized, using NumPy on CPU. Weight shapes are read
through safetensors slices. This module does not decide model compatibility.
Pipeline inspection only reads state; it never loads or removes adapters.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from safetensors import SafetensorError, safe_open

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline


_LORA_SUFFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("down", ".lora_down.weight"),
    ("up", ".lora_up.weight"),
    ("alpha", ".alpha"),
)
_TEXT_ENCODER_PREFIXES: Final[tuple[str, ...]] = (
    "lora_te_",
    "lora_te1_",
    "lora_te2_",
)


class LoRAInspectionError(Exception):
    """A checkpoint could not be accessed or read as safetensors."""


@dataclass(frozen=True)
class PipelineStateIssue:
    """A non-base state or a state that could not be safely inspected."""

    component: str
    state: str
    actual: str


@dataclass(frozen=True)
class PipelineStateReport:
    """Read-only pre-load diagnostics. Empty/None adapter state is clean."""

    issues: tuple[PipelineStateIssue, ...]

    @property
    def is_clean(self) -> bool:
        return not self.issues


def inspect_pipeline_state(pipe: StableDiffusionPipeline) -> PipelineStateReport:
    """Check pipeline, UNet and all present text encoders without mutation.

    Uses Diffusers/Transformers read-only getters, PEFT configs/tuner state,
    and exposed fusion markers. A disabled adapter is still attached. Fusion
    whose tracking was manually erased cannot be inferred from weight shapes.
    No torch, diffusers or peft import is needed to inspect these attributes.
    """
    issues: list[PipelineStateIssue] = []
    if pipe is None:
        return PipelineStateReport((PipelineStateIssue("pipeline", "missing", "None"),))

    _inspect_adapter_state(pipe, "pipeline", issues)
    _inspect_adapter_getter(pipe, "pipeline", "get_active_adapters", issues)
    # Read component configs directly instead of get_list_adapters(), which
    # assumes peft_config is a dict and cannot handle a None config safely.

    for component in ("unet", "text_encoder", "text_encoder_2"):
        root = getattr(pipe, component, None)
        if root is None:
            if component != "text_encoder_2":
                issues.append(
                    PipelineStateIssue(component, "missing_base_component", "None")
                )
            continue
        named_modules = getattr(root, "named_modules", None)
        if not callable(named_modules):
            issues.append(
                PipelineStateIssue(component, "invalid_base_component", type(root).__name__)
            )
            continue
        for name, module in named_modules():
            location = f"{component}.{name}" if name else component
            _inspect_adapter_state(module, location, issues)
            # Attention processors in older Diffusers versions need not be
            # registered as nn.Modules, so named_modules() alone is insufficient.
            processor = getattr(module, "processor", None)
            if processor is not None:
                _inspect_adapter_state(processor, f"{location}.processor", issues)
                if type(processor).__name__.startswith("LoRA"):
                    issues.append(
                        PipelineStateIssue(location, "lora_processor", type(processor).__name__)
                    )

    return PipelineStateReport(tuple(issues))


def _has_adapter_state(value: object) -> bool:
    """Avoid tensor truth-value evaluation and accept empty containers/None."""
    if value is None:
        return False
    if isinstance(value, (bool, int, float)):
        return bool(value)
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except TypeError:
        return True


def _inspect_adapter_state(
    owner: object, component: str, issues: list[PipelineStateIssue]
) -> None:
    for attribute in (
        "peft_config",
        "_hf_peft_config_loaded",
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
        "lora_layer",
        "merged_adapters",
        "merged",
        "num_fused_loras",
        # Legacy LoRACompatibleLinear/Conv retain these after fusion.
        "w_up",
        "w_down",
    ):
        value = getattr(owner, attribute, None)
        if _has_adapter_state(value):
            if isinstance(value, (bool, int, float, str, list, tuple)):
                actual = repr(value)
            elif callable(getattr(value, "keys", None)):
                actual = repr(tuple(value.keys()))
            else:
                actual = type(value).__name__
            issues.append(PipelineStateIssue(component, attribute, actual))

    if any(
        base.__name__ == "BaseTunerLayer" and base.__module__.startswith("peft.")
        for base in type(owner).__mro__
    ):
        issues.append(
            PipelineStateIssue(component, "injected_adapter_layer", type(owner).__name__)
        )

    for attribute in ("active_adapters", "active_adapter"):
        _inspect_adapter_getter(owner, component, attribute, issues)


def _inspect_adapter_getter(
    owner: object, component: str, attribute: str, issues: list[PipelineStateIssue]
) -> None:
    value = getattr(owner, attribute, None)
    if callable(value):
        # Fresh Diffusers/Transformers models expose this method but raise
        # "No adapter loaded" when this flag is false. Do not call it then.
        if attribute in ("active_adapters", "active_adapter") and getattr(
            owner, "_hf_peft_config_loaded", None
        ) is False:
            return
        try:
            value = value()
        except ValueError as exc:
            # Optional Diffusers getters require a PEFT backend even when no
            # adapter exists. Component/module checks still run without PEFT.
            if attribute.startswith("get_") and "PEFT backend is required" in str(exc):
                return
            issues.append(PipelineStateIssue(component, f"{attribute}_unverified", str(exc)))
            return
        except Exception as exc:
            issues.append(PipelineStateIssue(component, f"{attribute}_unverified", str(exc)))
            return
    if _has_adapter_state(value):
        issues.append(PipelineStateIssue(component, attribute, repr(value)))


@dataclass
class LoRAInspectionReport:
    """Checkpoint contents, with sorted unique ranks and alpha values.

    Component counts include weights, alphas, and any unrecognized tensors.
    Pairing compares module names, without validating weight compatibility.
    Unexpected keys have an unknown component/suffix, a down weight whose rank
    cannot be inferred, or an alpha tensor that does not contain one value.
    """

    path: Path
    file_size_bytes: int
    metadata: dict[str, str]
    tensor_count: int
    unet_tensor_count: int
    text_encoder_tensor_count: int
    other_tensor_count: int
    lora_down_count: int
    lora_up_count: int
    alpha_count: int
    ranks: tuple[int, ...]
    alphas: tuple[float, ...]
    unmatched_down_keys: tuple[str, ...]
    unmatched_up_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def inspect_lora(checkpoint_path: str | Path) -> LoRAInspectionReport:
    """Inspect exactly one checkpoint, raising a path-specific error on failure.

    Supports sd-scripts keys prefixed by lora_unet_, lora_te_, lora_te1_, or
    lora_te2_. Missing metadata/alphas, unpaired weights, and unknown keys are
    reported without rejecting the checkpoint. The returned path is absolute.
    """
    path = Path(checkpoint_path).expanduser()
    try:
        path = path.resolve()
        file_size_bytes = _check_checkpoint_file(path)
        with safe_open(str(path), framework="np", device="cpu") as checkpoint:
            return _inspect_checkpoint(checkpoint, path, file_size_bytes)
    except (OSError, SafetensorError, ValueError, TypeError) as exc:
        raise LoRAInspectionError(
            f"Could not inspect LoRA checkpoint '{path}': {exc}"
        ) from exc


def _check_checkpoint_file(path: Path) -> int:
    if not path.exists():
        raise LoRAInspectionError(f"LoRA checkpoint does not exist: {path}")
    if not path.is_file():
        raise LoRAInspectionError(f"LoRA checkpoint path must be a file: {path}")
    if path.suffix.lower() != ".safetensors":
        raise LoRAInspectionError(
            f"LoRA checkpoint must use the .safetensors extension: {path}"
        )
    return path.stat().st_size


def _classify_tensor_key(key: str) -> Literal["unet", "text_encoder", "other"]:
    if key.startswith("lora_unet_"):
        return "unet"
    if key.startswith(_TEXT_ENCODER_PREFIXES):
        return "text_encoder"
    return "other"


def _split_lora_key(key: str) -> tuple[str, str] | None:
    for role, suffix in _LORA_SUFFIXES:
        if key.endswith(suffix):
            module = key[: -len(suffix)]
            if module:
                return module, role
    return None


def _inspect_checkpoint(
    checkpoint: safe_open, path: Path, file_size_bytes: int
) -> LoRAInspectionReport:
    keys = tuple(sorted(checkpoint.keys()))
    if not keys:
        raise LoRAInspectionError(f"LoRA checkpoint contains no tensors: {path}")
    metadata = dict(checkpoint.metadata() or {})
    component_counts: Counter[str] = Counter()
    down_keys: dict[str, str] = {}
    up_keys: dict[str, str] = {}
    alpha_count = 0
    ranks: set[int] = set()
    alphas: set[float] = set()
    unexpected_keys: set[str] = set()

    for key in keys:
        component = _classify_tensor_key(key)
        component_counts[component] += 1
        if component == "other":
            unexpected_keys.add(key)

        module_role = _split_lora_key(key)
        if module_role is None:
            unexpected_keys.add(key)
            continue
        module, role = module_role

        if role == "down":
            down_keys[module] = key
            shape = checkpoint.get_slice(key).get_shape()
            if len(shape) >= 2:
                ranks.add(shape[0])
            else:
                unexpected_keys.add(key)
        elif role == "up":
            up_keys[module] = key
        else:
            alpha_count += 1
            shape = checkpoint.get_slice(key).get_shape()
            if prod(shape) == 1:
                alphas.add(float(checkpoint.get_tensor(key).item()))
            else:
                unexpected_keys.add(key)

    return LoRAInspectionReport(
        path=path,
        file_size_bytes=file_size_bytes,
        metadata=metadata,
        tensor_count=len(keys),
        unet_tensor_count=component_counts["unet"],
        text_encoder_tensor_count=component_counts["text_encoder"],
        other_tensor_count=component_counts["other"],
        lora_down_count=len(down_keys),
        lora_up_count=len(up_keys),
        alpha_count=alpha_count,
        ranks=tuple(sorted(ranks)),
        alphas=tuple(sorted(alphas)),
        unmatched_down_keys=tuple(
            sorted(down_keys[module] for module in down_keys.keys() - up_keys.keys())
        ),
        unmatched_up_keys=tuple(
            sorted(up_keys[module] for module in up_keys.keys() - down_keys.keys())
        ),
        unexpected_keys=tuple(sorted(unexpected_keys)),
    )
