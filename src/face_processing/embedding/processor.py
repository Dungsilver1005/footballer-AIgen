"""Persist an in-memory aligned face embedding without modifying Stage 1/2 files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from ..detection.processor import DEFAULT_OUTPUT_DIR
from .embedder import ArcFaceEmbedder, resolve_project_path
from .validator import INPUT_TENSOR_SHAPE, validate_crop_metadata, validate_normalized_embedding


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: NDArray[np.float32]
    output_path: Path
    metadata_path: Path


def load_stage1_crop(
    crop_path: str | Path, metadata_path: str | Path | None = None
) -> tuple[NDArray[np.uint8], dict]:
    """Read only a Stage 1 <source filename>.face.png RGB crop and its JSON.

    Return pixels and validated metadata. Inference and alignment are left to
    the caller. Manual *_phase2_aligned.png outputs are not accepted as crops.
    """
    crop_path = resolve_project_path(crop_path)
    if not crop_path.name.endswith(".face.png"):
        raise ValueError("Stage 1 crop filename must end with .face.png")
    metadata_path = (
        crop_path.with_suffix(".json")
        if metadata_path is None else resolve_project_path(metadata_path)
    )
    try:
        with Image.open(crop_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Stage 1 crop must be a single-frame RGB PNG")
            image.load()
            crop = np.array(image, dtype=np.uint8)
            crop_size = image.size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot read Stage 1 crop {crop_path}: {exc}") from exc
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read Stage 1 metadata {metadata_path}: {exc}") from exc
    validate_crop_metadata(metadata, crop_size)
    # PureWindowsPath also handles Stage 1 metadata moved from a Windows host.
    source_name = PureWindowsPath(metadata["source_image"]).name
    if crop_path.name != f"{source_name}.face.png":
        raise ValueError("Stage 1 source_image filename does not match the crop filename")
    return crop, metadata


def process_aligned_face(
    aligned_face_rgb: NDArray[np.uint8],
    embedder: ArcFaceEmbedder,
    source_image: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    crop_image: str | Path | None = None,
    transform_matrix: ArrayLike | None = None,
) -> EmbeddingResult:
    """Extract and save one embedding from RGB pixels already aligned in memory.

    The embedder must already be loaded. Raise on invalid input, inference or
    output errors. Exclusive creation preserves existing NPY/JSON files; only
    files created by this call are removed if either write fails.
    """
    if not isinstance(source_image, (str, Path)) or not str(source_image).strip():
        raise ValueError("source_image must be a non-empty image path")
    source_path = resolve_project_path(source_image)
    crop_path = None if crop_image is None else resolve_project_path(crop_image)
    if crop_path is not None and crop_path.name != f"{source_path.name}.face.png":
        raise ValueError("crop_image must be named <source filename>.face.png")
    output_dir = resolve_project_path(output_dir)
    stem = crop_path.stem if crop_path is not None else f"{source_path.name}.face"
    output_path = output_dir / f"{stem}.embedding.npy"
    metadata_path = output_dir / f"{stem}.embedding.json"
    # Also refuse to create a target that is an explicitly supplied input path.
    if any(path in (source_path, crop_path) for path in (output_path, metadata_path)):
        raise ValueError("Embedding outputs must not replace source or crop images")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Embedding output already exists: {output_path} or {metadata_path}")

    matrix = None
    if transform_matrix is not None:
        try:
            matrix = np.asarray(transform_matrix)
        except (TypeError, ValueError) as exc:
            raise ValueError("transform_matrix must be a finite real 2x3 array") from exc
        if matrix.shape != (2, 3) or matrix.dtype.kind not in "iuf" or not np.isfinite(matrix).all():
            raise ValueError("transform_matrix must be a finite real 2x3 array")
        matrix = matrix.tolist()

    embedding = embedder.extract_embedding(aligned_face_rgb)
    validate_normalized_embedding(embedding)
    metadata = {
        "source_image": str(source_path),
        "crop_image": None if crop_path is None else str(crop_path),
        "model_name": embedder.model_path.name,
        "model_path": str(embedder.model_path),
        "input_shape": list(INPUT_TENSOR_SHAPE),
        "aligned_image_shape": list(aligned_face_rgb.shape),
        "embedding_dimension": int(embedding.size),
        "dtype": str(embedding.dtype),
        "normalization": "l2",
        "input_normalization": {
            "location": "onnx_graph" if embedder.input_mean == 0.0 else "python",
            "mean": embedder.input_mean,
            "std": embedder.input_std,
            "channel_order": "RGB",
        },
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
    }
    if matrix is not None:
        metadata["transform_matrix"] = matrix
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)

    created_paths: list[Path] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as output_file:
            created_paths.append(output_path)
            np.save(output_file, embedding, allow_pickle=False)
        with metadata_path.open("x", encoding="utf-8") as metadata_file:
            created_paths.append(metadata_path)
            metadata_file.write(metadata_text + "\n")
    except Exception:
        for created_path in reversed(created_paths):
            try:
                created_path.unlink(missing_ok=True)
            except OSError as exc:
                # Preserve the write error; report any file that could not be removed.
                logger.error("Partial output cleanup failed: %s: %s", created_path, exc)
        raise
    logger.info("Saved ArcFace embedding: %s; metadata: %s", output_path, metadata_path)
    return EmbeddingResult(embedding, output_path, metadata_path)
