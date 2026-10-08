"""Scheduler selection for the local WD 1.5 Beta3 inference pipeline."""

from typing import Any, Type, Union

from .config import SchedulerConfig


_SUPPORTED_SCHEDULERS = ("euler", "euler_a", "ddim", "dpm_solver")
_SCHEDULER_ALIASES = {
    "euler": "euler",
    "euler_discrete": "euler",
    "euler_a": "euler_a",
    "euler_ancestral": "euler_a",
    "euler_ancestral_discrete": "euler_a",
    "ddim": "ddim",
    "dpm_solver": "dpm_solver",
    "dpm_solver_multistep": "dpm_solver",
    "dpmsolver": "dpm_solver",
}


def configure_scheduler(
    pipe: Any, scheduler: Union[SchedulerConfig, str]
) -> Any:
    """Replace ``pipe.scheduler`` with the requested scheduler and return ``pipe``."""
    scheduler_name = (
        scheduler.name if isinstance(scheduler, SchedulerConfig) else scheduler
    )
    if not isinstance(scheduler_name, str):
        raise ValueError(
            "Scheduler name must be a string. Supported schedulers: "
            f"{', '.join(_SUPPORTED_SCHEDULERS)}."
        )

    normalized_name = (
        scheduler_name.strip().lower().replace("-", "_").replace(" ", "_")
    )
    canonical_name = _SCHEDULER_ALIASES.get(normalized_name)
    if canonical_name is None:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. Supported schedulers: "
            f"{', '.join(_SUPPORTED_SCHEDULERS)}."
        )

    scheduler_config = pipe.scheduler.config

    # Import lazily so this module can be imported without loading Diffusers.
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
    )

    scheduler_classes: dict[str, Type[Any]] = {
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "ddim": DDIMScheduler,
        "dpm_solver": DPMSolverMultistepScheduler,
    }
    new_scheduler = scheduler_classes[canonical_name].from_config(scheduler_config)
    pipe.scheduler = new_scheduler
    return pipe
