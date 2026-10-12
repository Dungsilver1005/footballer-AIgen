"""Validate still images, face counts and crop coordinates."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from PIL import Image, ImageOps

if TYPE_CHECKING:
    from .detector import FaceDetection


SUPPORTED_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    INVALID_IMAGE = "invalid_image"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    INVALID_BBOX = "invalid_bbox"
    DETECTION_ERROR = "detection_error"
    OUTPUT_ERROR = "output_error"


def read_valid_image(image_path: str | Path) -> Image.Image:
    """Return a fully decoded RGB image with EXIF orientation applied.

    Check both the extension and the decoded format. Animated or multipage
    images are outside this still-image pipeline. No image is written here.
    """
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    try:
        with Image.open(path) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {image.format}")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Only single-frame still images are supported")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot read image {path}: {exc}") from exc


def validate_face_count(detections: Sequence[FaceDetection]) -> ProcessingStatus:
    """Require exactly one face; never choose one face from a group."""
    if not detections:
        return ProcessingStatus.NO_FACE
    if len(detections) > 1:
        return ProcessingStatus.MULTIPLE_FACES
    return ProcessingStatus.SUCCESS


def validate_bbox(
    bbox: Sequence[float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Validate xyxy geometry and its intersection with a (width, height) image.

    Partially out-of-bounds boxes are accepted and clipped during cropping.
    Empty, reversed, non-finite and fully out-of-bounds boxes are rejected.
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain four xyxy coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("Bounding box coordinates must be finite")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box must have positive width and height")
    if min(x2, width) <= max(x1, 0) or min(y2, height) <= max(y1, 0):
        raise ValueError("Bounding box does not intersect the image")
    return x1, y1, x2, y2


def validate_margin(margin: float) -> None:
    """Margin is a finite, non-negative fraction of each bbox dimension."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Crop margin must be finite and non-negative")
