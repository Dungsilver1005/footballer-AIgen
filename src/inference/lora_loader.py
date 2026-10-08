"""Attach one unfused LoRA only after validation of a clean WD Beta3 base.

The validator owns checkpoint inspection and compatibility. This module applies
its approved in-memory conversions, loads weights, and verifies the adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from safetensors.torch import load_file

from .lora_validator import (
    LoRAValidationError,
    LoRAValidationReport,
    validate_lora,
)

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline
    from torch import Tensor


class LoRALoadError(Exception):
    """A validation, loading or verification failure with retained diagnostics.

    After an actual load attempt, pipeline state may have changed even if
    loading raised. No rollback is attempted. Validator exceptions remain
    accessible through __cause__, including their original diagnostic code.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        checkpoint_path: str | Path,
        validation_report: LoRAValidationReport | None = None,
        adapter_name: str | None = None,
        pipeline_state_may_have_changed: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.checkpoint_path = checkpoint_path
        self.validation_report = validation_report
        self.adapter_name = adapter_name
        self.pipeline_state_may_have_changed = pipeline_state_may_have_changed


@dataclass(frozen=True)
class LoRALoadResult:
    """Successful attachment, including the resolved path and adapter name.

    The complete validation report retains its errors, warnings and inspection
    metadata. The caller's pipeline is mutated in place and is no longer clean.
    """

    checkpoint_path: Path
    validation_report: LoRAValidationReport
    adapter_name: str
    loaded: bool


def load_lora(
    pipe: StableDiffusionPipeline,
    checkpoint_path: str | Path,
    *,
    adapter_name: str | None = None,
) -> LoRALoadResult:
    """Validate, load and verify one local sd-scripts LoRA on an existing base.

    Validation is mandatory and runs on the caller's original pipeline state.
    An incompatible report or validator exception prevents any load operation;
    warnings alone do not reject a compatible checkpoint. None lets Diffusers
    choose the adapter name, which is read back during verification.

    Uses load_lora_weights with an explicit local safetensors filename, or a
    locally prepared state dict when validated Linear targets require 1x1
    weight conversion. No fusion, adapter switching, device/dtype changes or
    generation is requested.
    Load and post-load verification failures raise LoRALoadError and may leave
    a partially modified pipeline; they do not trigger automatic cleanup.
    """
    try:
        validation_report = validate_lora(checkpoint_path, pipe)
    except LoRAValidationError as exc:
        raise LoRALoadError(
            f"LoRA validation failed for '{checkpoint_path}': {exc}",
            code="validation_failed",
            checkpoint_path=checkpoint_path,
            adapter_name=adapter_name,
        ) from exc

    path = validation_report.inspection_report.path
    if not validation_report.is_compatible:
        details = "; ".join(
            f"{issue.code}: {issue.reason}" for issue in validation_report.errors
        ) or "No LoRA modules passed validation."
        raise LoRALoadError(
            f"LoRA validation rejected '{path}': {details}",
            code="validation_rejected",
            checkpoint_path=path,
            validation_report=validation_report,
            adapter_name=adapter_name,
        )

    if adapter_name is not None and (
        not isinstance(adapter_name, str) or not adapter_name.strip()
    ):
        raise LoRALoadError(
            f"Invalid adapter name for '{path}': expected a nonempty string or None.",
            code="invalid_adapter_name",
            checkpoint_path=path,
            validation_report=validation_report,
            adapter_name=adapter_name,
        )

    # Finish every approved conversion before the only attachment operation.
    state_dict = None
    if validation_report.conversion_plan:
        state_dict = _prepare_converted_state_dict(
            path, validation_report, adapter_name
        )

    try:
        if state_dict is None:
            pipe.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=adapter_name,
                local_files_only=True,
                use_safetensors=True,
            )
        else:
            pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
    except Exception as exc:
        raise LoRALoadError(
            f"LoRA loading failed for '{path}': {exc}; "
            "pipeline state may have changed. No rollback was attempted.",
            code="loading_failed",
            checkpoint_path=path,
            validation_report=validation_report,
            adapter_name=adapter_name,
            pipeline_state_may_have_changed=True,
        ) from exc

    try:
        loaded_adapter_name = _verify_adapter_attached(
            pipe, validation_report, adapter_name
        )
    except Exception as exc:
        raise LoRALoadError(
            f"LoRA post-load verification failed for '{path}': {exc}; "
            "pipeline state may have changed. No rollback was attempted.",
            code="post_load_verification_failed",
            checkpoint_path=path,
            validation_report=validation_report,
            adapter_name=adapter_name,
            pipeline_state_may_have_changed=True,
        ) from exc

    return LoRALoadResult(
        checkpoint_path=path,
        validation_report=validation_report,
        adapter_name=loaded_adapter_name,
        loaded=True,
    )


def _prepare_converted_state_dict(
    path: Path,
    validation_report: LoRAValidationReport,
    adapter_name: str | None,
) -> dict[str, Tensor]:
    """Read local weights and apply only conversions named by validation."""
    try:
        state_dict = dict(load_file(str(path), device="cpu"))
    except Exception as exc:
        raise LoRALoadError(
            f"LoRA conversion could not read '{path}': {exc}",
            code="conversion_failed",
            checkpoint_path=path,
            validation_report=validation_report,
            adapter_name=adapter_name,
        ) from exc

    for conversion in validation_report.conversion_plan:
        tensor = state_dict.get(conversion.key)
        actual_shape = tuple(tensor.shape) if tensor is not None else "missing"
        if (
            conversion.kind != "conv1x1_to_linear"
            or len(conversion.source_shape) != 4
            or conversion.source_shape[2:] != (1, 1)
            or conversion.target_shape != conversion.source_shape[:2]
            or actual_shape != conversion.source_shape
        ):
            raise LoRALoadError(
                f"LoRA conversion failed for '{conversion.key}' "
                f"({conversion.kind}): expected source shape "
                f"{conversion.source_shape}, actual {actual_shape}; "
                f"expected target shape {conversion.target_shape}.",
                code="conversion_failed",
                checkpoint_path=path,
                validation_report=validation_report,
                adapter_name=adapter_name,
            )
        try:
            state_dict[conversion.key] = tensor.squeeze(-1).squeeze(-1)
        except Exception as exc:
            raise LoRALoadError(
                f"LoRA conversion failed for '{conversion.key}' "
                f"({conversion.kind}): expected source shape "
                f"{conversion.source_shape}, actual {actual_shape}: {exc}",
                code="conversion_failed",
                checkpoint_path=path,
                validation_report=validation_report,
                adapter_name=adapter_name,
            ) from exc
    return state_dict


def _verify_adapter_attached(
    pipe: StableDiffusionPipeline,
    validation_report: LoRAValidationReport,
    adapter_name: str | None,
) -> str:
    """Read registration and injected layers in each checkpoint component.

    Direct peft_config reads also support text-encoder-only LoRAs and untouched
    components with peft_config=None. No clean-pipeline check is run after load.
    This verifies attachment and activation, not tensor values or image quality.
    """
    inspection = validation_report.inspection_report
    resolved_name = adapter_name
    for component_name, tensor_count in (
        ("unet", inspection.unet_tensor_count),
        ("text_encoder", inspection.text_encoder_tensor_count),
    ):
        if not tensor_count:
            continue
        component = getattr(pipe, component_name, None)
        configs = getattr(component, "peft_config", None)
        if not isinstance(configs, Mapping) or not configs:
            raise ValueError(f"pipe.{component_name} has no registered PEFT adapter.")

        if resolved_name is None:
            names = tuple(configs)
            if len(names) != 1 or not isinstance(names[0], str):
                raise ValueError(
                    f"Cannot identify one loaded adapter in pipe.{component_name}: {names}."
                )
            resolved_name = names[0]
        if resolved_name not in configs:
            raise ValueError(
                f"Adapter '{resolved_name}' is not registered in pipe.{component_name}."
            )

        attached = False
        for module_name, module in component.named_modules():
            has_down = resolved_name in getattr(module, "lora_A", {})
            has_up = resolved_name in getattr(module, "lora_B", {})
            if not (has_down or has_up):
                continue
            location = f"{component_name}.{module_name}"
            if not (has_down and has_up):
                raise ValueError(
                    f"Adapter '{resolved_name}' lacks a paired lora_A/lora_B in {location}."
                )
            if resolved_name not in getattr(module, "active_adapters", ()) or getattr(
                module, "disable_adapters", False
            ):
                raise ValueError(f"Adapter '{resolved_name}' is not active in {location}.")
            if getattr(module, "merged", False):
                raise ValueError(f"Adapter '{resolved_name}' is fused in {location}.")
            attached = True

        if not attached:
            raise ValueError(
                f"Adapter '{resolved_name}' has no injected LoRA layers in pipe.{component_name}."
            )

    if resolved_name is None:
        raise ValueError("No loaded adapter could be identified in the pipeline.")
    return resolved_name
