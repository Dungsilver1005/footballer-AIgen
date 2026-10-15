"""Small validation helpers for ArcFace tensors and Stage 1 metadata."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..detection.detector import LANDMARK_ORDER


EMBEDDING_DIMENSION = 512
ALIGNED_IMAGE_SHAPE = (112, 112, 3)
INPUT_TENSOR_SHAPE = (1, 3, 112, 112)


def validate_aligned_face(image: NDArray[np.uint8]) -> None:
    """Require RGB pixels supplied by the caller; never resize or swap channels."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.shape != ALIGNED_IMAGE_SHAPE
    ):
        raise ValueError("Expected an aligned RGB uint8 image with shape (112, 112, 3)")


def _accepts_single_batch(dimension: Any) -> bool:
    return (
        dimension is None
        or (isinstance(dimension, str) and bool(dimension.strip()))
        or (type(dimension) is int and dimension == 1)
    )


def validate_model_io(inputs: Sequence[Any], outputs: Sequence[Any]) -> tuple[str, str]:
    """Validate ORT NodeArgs; only the batch axis may be symbolic/unknown."""
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("ArcFace requires exactly one ONNX input and one output")
    for item, label, expected_tail in (
        (inputs[0], "input", (3, 112, 112)),
        (outputs[0], "output", (EMBEDDING_DIMENSION,)),
    ):
        if not isinstance(item.name, str) or not item.name.strip():
            raise ValueError(f"ArcFace {label} must have a non-empty name")
        if item.type != "tensor(float)":
            raise ValueError(f"ArcFace {label} must have float32 dtype (tensor(float))")
        shape = item.shape
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != len(expected_tail) + 1
            or not _accepts_single_batch(shape[0])
            or any(type(value) is not int for value in shape[1:])
            or tuple(shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"ArcFace {label} shape must be [1 or dynamic, "
                f"{', '.join(str(value) for value in expected_tail)}]; got {shape}"
            )
    return inputs[0].name, outputs[0].name


def validate_raw_embedding(embedding: NDArray[np.floating]) -> float:
    """Require a finite floating 512-vector and return its positive L2 norm."""
    if (
        not isinstance(embedding, np.ndarray)
        or embedding.shape != (EMBEDDING_DIMENSION,)
        or embedding.dtype.kind != "f"
    ):
        raise ValueError("Raw embedding must be a floating array with shape (512,)")
    if not np.isfinite(embedding).all():
        raise ValueError("Raw embedding must not contain NaN or Inf")
    # float64 avoids overflow/underflow for finite float32 model outputs.
    norm = float(np.linalg.norm(embedding.astype(np.float64)))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Raw embedding must have a finite L2 norm greater than zero")
    return norm


def validate_normalized_embedding(embedding: NDArray[np.float32]) -> None:
    norm = validate_raw_embedding(embedding)
    if embedding.dtype != np.float32:
        raise ValueError("Normalized embedding must have float32 dtype")
    if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Normalized embedding must have L2 norm approximately 1; got {norm}")


def validate_crop_metadata(
    metadata: Any, crop_size: tuple[int, int]
) -> NDArray[np.float64]:
    """Return five crop-space points; crop_size is Pillow's (width, height).

    Stage 1 may retain landmarks outside the cropped rectangle, so finite
    coordinates are not clipped or rejected merely for crossing its bounds.
    Stage 2 checks whether their geometry admits a similarity transform.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Stage 1 metadata must be a JSON object")
    source = metadata.get("source_image")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Stage 1 metadata requires source_image")
    if metadata.get("landmark_order") != list(LANDMARK_ORDER):
        raise ValueError(f"Stage 1 landmark_order must be {list(LANDMARK_ORDER)}")
    size = metadata.get("crop_size")
    if (
        not isinstance(size, dict)
        or any(type(size.get(key)) is not int or size[key] <= 0 for key in ("width", "height"))
        or (size["width"], size["height"]) != crop_size
    ):
        raise ValueError(f"Stage 1 crop_size must match crop image size {crop_size}")
    try:
        points = np.asarray(metadata.get("landmarks_in_crop"))
    except (TypeError, ValueError) as exc:
        raise ValueError("landmarks_in_crop must contain five numeric (x, y) pairs") from exc
    if points.shape != (5, 2) or points.dtype.kind not in "iuf":
        raise ValueError("landmarks_in_crop must contain five real numeric (x, y) pairs")
    points = points.astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError("landmarks_in_crop coordinates must be finite")
    return points
