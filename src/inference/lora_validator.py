"""Validate sd-scripts LoRA shapes against an already loaded pipeline.

No adapters are loaded and no pipeline state is changed. Inspection reads only
alpha values on CPU; this module reads tensor shapes through safetensors slices.
Compatibility requires a clean, verified WD Beta3 base and valid module/shapes;
it does not predict generated quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Final

from safetensors import safe_open
from torch import nn

from .config import BaseModelProvenance, WD_BETA3_SHA256
from .lora_inspector import (
    LoRAInspectionReport,
    _split_lora_key,
    inspect_lora,
    inspect_pipeline_state,
)

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline


_COMPONENT_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("lora_unet_", "unet"),
    ("lora_te_", "text_encoder"),
    ("lora_te1_", "text_encoder"),
    ("lora_te2_", "text_encoder_2"),
)
_Shape = tuple[int, ...]
_IssueValue = str | int | tuple[int, ...] | tuple[float, ...] | tuple[str, ...]
_ModuleTensors = dict[str, tuple[str, _Shape]]
_ModuleMap = dict[str, list[tuple[str, nn.Module]]]


class LoRAValidationError(Exception):
    """The checkpoint could not be read or validation could not operate."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LoRAValidationIssue:
    """One diagnostic; module is the full sd-scripts LoRA module identifier."""

    code: str
    reason: str
    component: str | None = None
    module: str | None = None
    key: str | None = None
    expected: _IssueValue | None = None
    actual: _IssueValue | None = None


@dataclass(frozen=True)
class LoRATensorConversion:
    """An exact, validated weight conversion required before attachment."""

    key: str
    component: str
    module: str
    target_name: str
    role: str
    source_shape: _Shape
    target_shape: _Shape
    kind: str


@dataclass(frozen=True)
class LoRAValidationReport:
    """Structural compatibility with a clean, verified WD Beta3 pipeline.

    Matched counts uniquely resolved LoRA module identifiers, even if their
    tensors are invalid. Validated counts modules that passed every check.
    Inspection diagnostics are retained in inspection_report. Warnings alone
    do not make an otherwise valid checkpoint incompatible. conversion_plan
    names only weights from validated Linear-targeted 1x1 modules.
    """

    inspection_report: LoRAInspectionReport
    matched_module_count: int
    validated_module_count: int
    errors: tuple[LoRAValidationIssue, ...]
    warnings: tuple[LoRAValidationIssue, ...]
    conversion_plan: tuple[LoRATensorConversion, ...] = ()

    @property
    def is_compatible(self) -> bool:
        return not self.errors and self.validated_module_count > 0


def validate_lora(
    checkpoint_path: str | Path,
    pipe: StableDiffusionPipeline,
) -> LoRAValidationReport:
    """Require a clean WD Beta3 base, then inspect and validate one checkpoint.

    lora_te_ and lora_te1_ address text_encoder; lora_te2_ addresses
    text_encoder_2 and is incompatible with pipelines missing that component.
    Standard Linear and ungrouped Conv2d adapters are supported. Linear
    targets also accept paired Conv2d-style 1x1 weights with an explicit
    conversion plan. Conv2d down kernels must match the target, and up kernels
    must be (1, 1), as in the project's sd-scripts implementation.

    Read/operational failures raise LoRAValidationError with their cause.
    Pre-load failures raise with code pipeline_not_clean, base_model_unverified,
    or base_model_mismatch before reading LoRA tensors or matching modules.
    Readable but incompatible tensors produce errors in the returned report.
    """
    try:
        if pipe is None:
            raise LoRAValidationError("An already loaded pipeline is required.")
        state = inspect_pipeline_state(pipe)
        if not state.is_clean:
            details = "; ".join(
                f"{issue.component}.{issue.state}: actual {issue.actual}"
                for issue in state.issues
            )
            raise LoRAValidationError(
                f"pipeline_not_clean: expected loaded base components without "
                f"attached/active/fused adapters; {details}",
                code="pipeline_not_clean",
            )
        verify_base_model_identity(pipe)
        inspection = inspect_lora(checkpoint_path)
        return _validate_checkpoint(inspection, pipe)
    except LoRAValidationError:
        raise
    except Exception as exc:
        # Operational failures must remain distinct from a readable checkpoint
        # whose compatibility diagnostics are returned normally.
        raise LoRAValidationError(
            f"Could not validate LoRA checkpoint '{checkpoint_path}': {exc}"
        ) from exc


def verify_base_model_identity(pipe: StableDiffusionPipeline) -> BaseModelProvenance:
    """Require the loader's fingerprint of the exact project WD Beta3 file.

    Paths, model labels and architecture are not identity evidence. Do not
    rehash the source: it may have moved since loading, and the loader already
    fingerprinted it. Component binding rejects stale/copied provenance after
    a UNet or text encoder replacement. Trust assumes no manual weight edits.
    """
    info = getattr(pipe, "_project_base_model_info", None)
    if (
        not isinstance(info, BaseModelProvenance)
        or info.unet is None
        or info.text_encoder is None
        or not isinstance(info.checkpoint_path, Path)
        or not info.checkpoint_path.is_absolute()
        or not isinstance(info.sha256, str)
        or len(info.sha256) != 64
        or any(char not in "0123456789abcdef" for char in info.sha256.lower())
    ):
        raise LoRAValidationError(
            "base_model_unverified: pipeline requires loader-computed SHA-256 "
            "provenance from load_base_model(); expected exact WD 1.5 Beta3, "
            "actual missing or invalid source fingerprint.",
            code="base_model_unverified",
        )
    if info.unet is not getattr(pipe, "unet", None) or info.text_encoder is not getattr(
        pipe, "text_encoder", None
    ):
        raise LoRAValidationError(
            "base_model_unverified: unet/text_encoder no longer match the "
            "components recorded by the base loader; expected original components.",
            code="base_model_unverified",
        )
    if info.sha256.lower() != WD_BETA3_SHA256:
        raise LoRAValidationError(
            f"base_model_mismatch: expected WD 1.5 Beta3 SHA-256 {WD_BETA3_SHA256}; "
            f"actual {info.sha256} from '{info.checkpoint_path}'.",
            code="base_model_mismatch",
        )
    return info


def _validate_checkpoint(
    inspection: LoRAInspectionReport, pipe: StableDiffusionPipeline
) -> LoRAValidationReport:
    errors: list[LoRAValidationIssue] = []
    modules: dict[str, _ModuleTensors] = {}
    with safe_open(str(inspection.path), framework="np", device="cpu") as checkpoint:
        for key in sorted(checkpoint.keys()):
            # Share the inspector's suffix parsing instead of maintaining a
            # second checkpoint format parser. Only component resolution is new.
            module_role = _split_lora_key(key)
            if module_role is None:
                errors.append(
                    LoRAValidationIssue(
                        code="unsupported_tensor",
                        reason="Tensor is not a supported sd-scripts LoRA weight or alpha.",
                        component=_resolve_component(key)[0],
                        key=key,
                        expected=".lora_down.weight, .lora_up.weight or .alpha",
                        actual=key,
                    )
                )
                continue
            module, role = module_role
            shape = tuple(checkpoint.get_slice(key).get_shape())
            modules.setdefault(module, {})[role] = (key, shape)

    if not inspection.lora_down_count and not inspection.lora_up_count:
        errors.append(
            LoRAValidationIssue(
                code="no_lora_weights",
                reason="Checkpoint contains no LoRA down/up weights to validate.",
                expected="at least one complete LoRA module",
                actual=0,
            )
        )

    component_maps: dict[str, _ModuleMap | None] = {}
    matched_count = 0
    validated_count = 0
    conversion_plan: list[LoRATensorConversion] = []
    for module, tensors in sorted(modules.items()):
        component, encoded_name = _resolve_component(module)
        module_errors = _validate_tensor_pair(module, component, tensors)
        if component is None:
            module_errors.append(
                LoRAValidationIssue(
                    code="unsupported_component",
                    reason="LoRA module has an unsupported component prefix.",
                    module=module,
                    expected=tuple(prefix for prefix, _ in _COMPONENT_PREFIXES),
                    actual=module,
                )
            )
        else:
            if component not in component_maps:
                component_maps[component] = _map_component(pipe, component)
            module_map = component_maps[component]
            if module_map is None:
                module_errors.append(
                    LoRAValidationIssue(
                        code="missing_component",
                        reason=f"Pipeline does not contain component '{component}'.",
                        component=component,
                        module=module,
                        expected=component,
                        actual="missing",
                    )
                )
            else:
                targets = module_map.get(encoded_name, [])
                if not targets:
                    module_errors.append(
                        LoRAValidationIssue(
                            code="missing_target",
                            reason=f"No target in pipe.{component} matches '{module}'.",
                            component=component,
                            module=module,
                            expected="an existing module in the referenced component",
                            actual=encoded_name,
                        )
                    )
                elif len(targets) > 1:
                    module_errors.append(
                        LoRAValidationIssue(
                            code="ambiguous_target",
                            reason="Multiple real modules encode to the same sd-scripts name.",
                            component=component,
                            module=module,
                            expected="one unique target module",
                            actual=tuple(name for name, _ in targets),
                        )
                    )
                else:
                    matched_count += 1
                    target_name, target = targets[0]
                    target_errors, target_conversions = _validate_target(
                        module, component, target_name, target, tensors
                    )
                    module_errors.extend(target_errors)
                    if not module_errors:
                        validated_count += 1
                        conversion_plan.extend(target_conversions)
        errors.extend(module_errors)

    return LoRAValidationReport(
        inspection_report=inspection,
        matched_module_count=matched_count,
        validated_module_count=validated_count,
        errors=tuple(errors),
        warnings=tuple(_inspection_warnings(inspection)),
        conversion_plan=tuple(conversion_plan),
    )


def _resolve_component(module: str) -> tuple[str | None, str]:
    for prefix, component in _COMPONENT_PREFIXES:
        if module.startswith(prefix):
            return component, module[len(prefix) :]
    return None, module


def _map_component(pipe: StableDiffusionPipeline, component: str) -> _ModuleMap | None:
    root = getattr(pipe, component, None)
    if root is None:
        return None
    if not isinstance(root, nn.Module):
        raise LoRAValidationError(
            f"Pipeline component '{component}' must be a torch.nn.Module; "
            f"got {type(root).__name__}."
        )

    module_map: _ModuleMap = {}
    # Forward encoding preserves underscores, indices and exact component
    # ownership. Keep collisions rather than choosing an arbitrary target.
    for name, target in root.named_modules():
        if name:
            module_map.setdefault(name.replace(".", "_"), []).append((name, target))
    return module_map


def _validate_tensor_pair(
    module: str, component: str | None, tensors: _ModuleTensors
) -> list[LoRAValidationIssue]:
    errors: list[LoRAValidationIssue] = []
    for role in ("down", "up"):
        if role not in tensors:
            key = f"{module}.lora_{role}.weight"
            errors.append(
                LoRAValidationIssue(
                    code="missing_weight",
                    reason=f"LoRA module is missing its lora_{role} weight.",
                    component=component,
                    module=module,
                    key=key,
                    expected=key,
                    actual="missing",
                )
            )

    ranks: dict[str, int] = {}
    for role, (key, shape) in tensors.items():
        if role == "alpha":
            if prod(shape) != 1:
                errors.append(
                    LoRAValidationIssue(
                        code="invalid_alpha_shape",
                        reason="Alpha must contain exactly one value.",
                        component=component,
                        module=module,
                        key=key,
                        expected=1,
                        actual=prod(shape),
                    )
                )
            continue
        if len(shape) not in (2, 4):
            errors.append(
                LoRAValidationIssue(
                    code="invalid_tensor_dimensions",
                    reason="LoRA weights must be 2D for Linear or 4D for Conv2d.",
                    component=component,
                    module=module,
                    key=key,
                    expected="2D or 4D",
                    actual=shape,
                )
            )
        if len(shape) >= 2:
            rank = shape[0] if role == "down" else shape[1]
            ranks[role] = rank
            if rank <= 0:
                errors.append(
                    LoRAValidationIssue(
                        code="non_positive_rank",
                        reason=f"LoRA {role} rank must be positive; got {rank}.",
                        component=component,
                        module=module,
                        key=key,
                        expected="rank > 0",
                        actual=rank,
                    )
                )
    if "down" in ranks and "up" in ranks and ranks["down"] != ranks["up"]:
        errors.append(
            LoRAValidationIssue(
                code="rank_mismatch",
                reason=f"LoRA up rank: expected {ranks['down']} but got {ranks['up']}.",
                component=component,
                module=module,
                key=tensors["up"][0],
                expected=ranks["down"],
                actual=ranks["up"],
            )
        )
    return errors


def _validate_target(
    module: str,
    component: str,
    target_name: str,
    target: nn.Module,
    tensors: _ModuleTensors,
) -> tuple[list[LoRAValidationIssue], list[LoRATensorConversion]]:
    errors: list[LoRAValidationIssue] = []
    conversions: list[LoRATensorConversion] = []
    if isinstance(target, nn.Linear):
        dimensions = 2
        input_size = target.in_features
        output_size = target.out_features
    elif isinstance(target, nn.Conv2d):
        dimensions = 4
        input_size = target.in_channels
        output_size = target.out_channels
        if target.groups != 1:
            errors.append(
                LoRAValidationIssue(
                    code="unsupported_conv_groups",
                    reason="Standard sd-scripts Conv2d validation requires an ungrouped target.",
                    component=component,
                    module=module,
                    expected=1,
                    actual=target.groups,
                )
            )
    else:
        return (
            [
                LoRAValidationIssue(
                    code="unsupported_target_type",
                    reason=f"Target '{target_name}' is not a supported Linear or Conv2d.",
                    component=component,
                    module=module,
                    expected="torch.nn.Linear or torch.nn.Conv2d",
                    actual=type(target).__name__,
                )
            ],
            [],
        )

    if isinstance(target, nn.Linear) and all(role in tensors for role in ("down", "up")):
        down_dimensions = len(tensors["down"][1])
        up_dimensions = len(tensors["up"][1])
        if (
            down_dimensions in (2, 4)
            and up_dimensions in (2, 4)
            and down_dimensions != up_dimensions
        ):
            errors.append(
                LoRAValidationIssue(
                    code="target_type_mismatch",
                    reason=(
                        f"Target '{target_name}' requires matching 2D weights "
                        "or paired 4D 1x1 weights."
                    ),
                    component=component,
                    module=module,
                    expected=down_dimensions,
                    actual=up_dimensions,
                )
            )

    for role in ("down", "up"):
        if role not in tensors:
            continue
        key, shape = tensors[role]
        convertible_linear = isinstance(target, nn.Linear) and len(shape) == 4
        if convertible_linear and shape[2:] != (1, 1):
            errors.append(
                LoRAValidationIssue(
                    code="kernel_mismatch",
                    reason=(
                        f"Target '{target_name}' {role} Linear conversion kernel: "
                        f"expected (1, 1) but got {shape[2:]}."
                    ),
                    component=component,
                    module=module,
                    key=key,
                    expected=(1, 1),
                    actual=shape[2:],
                )
            )
            continue
        if len(shape) != dimensions and not convertible_linear:
            # Other dimensionalities already have a tensor diagnostic. This
            # specifically identifies a Linear/Conv2d mismatch on either side.
            if len(shape) in (2, 4):
                errors.append(
                    LoRAValidationIssue(
                        code="target_type_mismatch",
                        reason=(
                            f"{role} weight cannot target '{target_name}': expected "
                            f"{dimensions}D for {type(target).__name__} but got {len(shape)}D."
                        ),
                        component=component,
                        module=module,
                        key=key,
                        expected=dimensions,
                        actual=len(shape),
                    )
                )
            continue
        size = shape[1] if role == "down" else shape[0]
        expected_size = input_size if role == "down" else output_size
        dimension_name = "input" if role == "down" else "output"
        if size != expected_size:
            errors.append(
                LoRAValidationIssue(
                    code=f"{dimension_name}_dimension_mismatch",
                    reason=(
                        f"Target '{target_name}' {dimension_name} dimension: "
                        f"expected {expected_size} but got {size}."
                    ),
                    component=component,
                    module=module,
                    key=key,
                    expected=expected_size,
                    actual=size,
                )
            )
        if isinstance(target, nn.Conv2d):
            expected_kernel = tuple(target.kernel_size) if role == "down" else (1, 1)
            if shape[2:] != expected_kernel:
                errors.append(
                    LoRAValidationIssue(
                        code="kernel_mismatch",
                        reason=(
                            f"Target '{target_name}' {role} kernel: "
                            f"expected {expected_kernel} but got {shape[2:]}."
                        ),
                        component=component,
                        module=module,
                        key=key,
                        expected=expected_kernel,
                        actual=shape[2:],
                    )
                )
        elif convertible_linear:
            conversions.append(
                LoRATensorConversion(
                    key=key,
                    component=component,
                    module=module,
                    target_name=target_name,
                    role=role,
                    source_shape=shape,
                    target_shape=shape[:2],
                    kind="conv1x1_to_linear",
                )
            )
    return errors, conversions


def _inspection_warnings(inspection: LoRAInspectionReport) -> list[LoRAValidationIssue]:
    warnings: list[LoRAValidationIssue] = []
    if len(inspection.ranks) > 1:
        warnings.append(
            LoRAValidationIssue(
                code="multiple_ranks",
                reason="Checkpoint uses multiple ranks; each module is checked independently.",
                actual=inspection.ranks,
            )
        )
    if len(inspection.alphas) > 1:
        warnings.append(
            LoRAValidationIssue(
                code="multiple_alphas",
                reason="Checkpoint uses multiple alpha values; this alone is not incompatible.",
                actual=inspection.alphas,
            )
        )
    network_module = inspection.metadata.get("ss_network_module")
    if network_module is not None and network_module != "networks.lora":
        warnings.append(
            LoRAValidationIssue(
                code="unrecognized_network_metadata",
                reason="Network metadata is unfamiliar; compatibility uses actual tensor shapes.",
                expected="networks.lora",
                actual=network_module,
            )
        )
    return warnings
