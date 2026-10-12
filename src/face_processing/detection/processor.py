"""Explicitly process local still images into face crops and JSON metadata.

Example, to be run by the caller only after preparing dependencies and weights::

    detector = SCRFDDetector(PROJECT_ROOT / "models/face_detection/2.5g_bnkps.onnx")
    detector.load_model()
    results = process_directory(detector)

Default input_face/ and output_face/ directories are relative to project root.
Importing this module performs no model loading, inference or output writes.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .detector import LANDMARK_ORDER, FaceDetection, SCRFDDetector
from .validator import (
    SUPPORTED_EXTENSIONS,
    ProcessingStatus,
    read_valid_image,
    validate_bbox,
    validate_face_count,
    validate_margin,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection" / "2.5g_bnkps.onnx"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input_face"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_face"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessingResult:
    """Status plus original detections, including those from skipped groups."""

    input_path: Path
    status: ProcessingStatus
    detections: tuple[FaceDetection, ...] = ()
    crop_box: tuple[int, int, int, int] | None = None
    output_path: Path | None = None
    metadata_path: Path | None = None
    message: str = ""


def compute_crop_box(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    margin: float = 0.15,
) -> tuple[int, int, int, int]:
    """Expand by margin * bbox width/height on EACH side, then clip.

    The default adds 15% on each side (30% total before clipping).
    Floor/ceil preserve fractional bbox coverage. The result is a Pillow
    crop box with exclusive right/bottom edges and no out-of-image pixels.
    """
    validate_margin(margin)
    x1, y1, x2, y2 = validate_bbox(bbox, image_size)
    width, height = image_size
    horizontal_margin = (x2 - x1) * margin
    vertical_margin = (y2 - y1) * margin
    return (
        math.floor(max(0.0, x1 - horizontal_margin)),
        math.floor(max(0.0, y1 - vertical_margin)),
        math.ceil(min(float(width), x2 + horizontal_margin)),
        math.ceil(min(float(height), y2 + vertical_margin)),
    )


def process_image(
    image_path: str | Path,
    detector: SCRFDDetector,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    margin: float = 0.15,
) -> ProcessingResult:
    """Save only a valid single-face crop; return a status for skipped images.

    The detector must already be loaded. PNG crops preserve RGB pixels; JSON
    stores original bbox/landmarks/confidence, EXIF-normalized image dimensions,
    crop bounds and translated landmarks. No resizing/alignment is performed.
    Existing outputs are never overwritten. Both output names include the
    source extension, so e.g. player.jpg and player.png do not collide.
    """
    validate_margin(margin)
    path = Path(image_path)
    logger.info("Processing image: %s", path)
    try:
        image = read_valid_image(path)
    except (OSError, ValueError) as exc:
        logger.error(
            "INVALID_IMAGE: %s: %s", path, exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return ProcessingResult(path, ProcessingStatus.INVALID_IMAGE, message=str(exc))
    logger.debug("Input image size: %s: %dx%d", path.name, image.width, image.height)

    try:
        # Pillow supplies RGB; InsightFace/OpenCV SCRFD expects BGR.
        bgr_image = np.asarray(image, dtype=np.uint8)[:, :, ::-1].copy()
        detections = tuple(detector.detect(bgr_image))
    except Exception as exc:
        logger.error(
            "DETECTION_ERROR: %s: %s", path, exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return ProcessingResult(path, ProcessingStatus.DETECTION_ERROR, message=str(exc))

    logger.info("Detected %d face(s): %s", len(detections), path.name)
    for index, detection in enumerate(detections, start=1):
        logger.debug(
            "Face %d: confidence=%.6f bbox=%s", index,
            detection.confidence, detection.bbox,
        )
    status = validate_face_count(detections)
    if status != ProcessingStatus.SUCCESS:
        logger.warning("%s: %s: detected %d face(s)", status.name, path, len(detections))
        return ProcessingResult(
            path, status, detections, message=f"Expected one face; detected {len(detections)}"
        )
    detection = detections[0]
    try:
        crop_box = compute_crop_box(detection.bbox, image.size, margin)
    except (TypeError, ValueError, OverflowError) as exc:
        logger.error(
            "INVALID_BBOX: %s: %s", path, exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return ProcessingResult(
            path, ProcessingStatus.INVALID_BBOX, detections, message=str(exc)
        )

    output_dir = Path(output_dir)
    output_path = output_dir / f"{path.name}.face.png"
    metadata_path = output_dir / f"{path.name}.face.json"
    left, top, right, bottom = crop_box
    logger.debug(
        "Crop size: %s: %dx%d (box=%s, margin=%s)",
        path.name, right - left, bottom - top, crop_box, margin,
    )
    metadata = {
        "source_image": str(path),
        "source_size": {"width": image.width, "height": image.height},
        "coordinate_space": "exif_transposed_image_pixels",
        "bbox_format": "xyxy",
        "landmark_order": list(LANDMARK_ORDER),
        "landmark_left_right": "image_viewer",
        "detection": asdict(detection),
        "crop_box": list(crop_box),
        "crop_margin": margin,
        "crop_size": {"width": right - left, "height": bottom - top},
        "landmarks_in_crop": [[x - left, y - top] for x, y in detection.landmarks],
    }

    created_paths: list[Path] = []
    logger.debug("Output PNG: %s", output_path)
    logger.debug("Output metadata JSON: %s", metadata_path)
    try:
        # Serialize before writing either file; JSON rejects NaN/Infinity.
        metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as output_file:
            created_paths.append(output_path)
            image.crop(crop_box).save(output_file, format="PNG")
        with metadata_path.open("x", encoding="utf-8") as metadata_file:
            created_paths.append(metadata_path)
            metadata_file.write(metadata_text + "\n")
    except (OSError, TypeError, ValueError) as exc:
        logger.error(
            "OUTPUT_ERROR: %s: %s", path, exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        cleanup_errors = []
        # Remove only files created by this call, never pre-existing outputs.
        for created_path in created_paths:
            try:
                created_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                logger.error(
                    "Partial output cleanup failed: %s: %s", created_path, cleanup_exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                cleanup_errors.append(str(cleanup_exc))
        message = str(exc)
        if cleanup_errors:
            message += "; Partial output cleanup failed: " + "; ".join(cleanup_errors)
        return ProcessingResult(
            path, ProcessingStatus.OUTPUT_ERROR, detections, crop_box, message=message
        )
    logger.info("SUCCESS: %s -> PNG: %s; metadata JSON: %s", path, output_path, metadata_path)
    return ProcessingResult(
        path, ProcessingStatus.SUCCESS, detections, crop_box, output_path, metadata_path
    )


def process_directory(
    detector: SCRFDDetector,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    margin: float = 0.15,
) -> list[ProcessingResult]:
    """Process supported files, non-recursively and in deterministic name order.

    Non-image extensions are ignored. Unreadable images and skipped detections
    produce per-file results without stopping the batch. Invalid configuration
    or a missing input directory raises an exception; no directory is created
    until a valid crop is ready to save.
    """
    validate_margin(margin)
    input_dir = Path(input_dir)
    logger.info("Input directory: %s; output directory: %s", input_dir, output_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input image directory does not exist: {input_dir}")
    image_paths = sorted(
        (
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name,
    )
    logger.info("Found %d supported image file(s)", len(image_paths))
    if not image_paths:
        logger.warning("No supported images found in %s", input_dir)
    return [process_image(path, detector, output_dir, margin=margin) for path in image_paths]
