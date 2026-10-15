r"""Run Stage 3 on a Stage 1 crop/JSON pair, without running detection again.

From project root::

    python -m src.face_processing.embedding --crop output_face/anh2.jpg.face.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from ..alignment import align_face
from ..detection.processor import DEFAULT_OUTPUT_DIR
from .embedder import DEFAULT_MODEL_PATH, ArcFaceEmbedder, resolve_project_path
from .processor import load_stage1_crop, process_aligned_face


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align a Stage 1 RGB crop, then extract a CPU-only ArcFace embedding.",
        epilog="Relative paths use project root. Existing outputs are preserved.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Local ArcFace ONNX model")
    parser.add_argument("--crop", type=Path, required=True,
                        help="Stage 1 <source filename>.face.png RGB crop")
    parser.add_argument("--metadata", type=Path,
                        help="Stage 1 JSON (default: crop filename with .json suffix)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for embedding NPY and JSON files")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    stage = "INPUT_ERROR"
    try:
        crop_path = resolve_project_path(args.crop)
        crop, metadata = load_stage1_crop(crop_path, args.metadata)
        stage = "ALIGNMENT_ERROR"
        aligned_face, transform_matrix = align_face(crop, metadata["landmarks_in_crop"])
        stage = "MODEL_LOAD_ERROR"
        embedder = ArcFaceEmbedder(args.model)
        embedder.load_model()
        stage = "EMBEDDING_ERROR"
        result = process_aligned_face(
            aligned_face, embedder, metadata["source_image"], args.output,
            crop_image=crop_path, transform_matrix=transform_matrix,
        )
    except Exception as exc:
        logger.error("%s: %s", stage, exc, exc_info=args.verbose)
        return 1
    print(
        f"Embedding: shape={result.embedding.shape}, dtype={result.embedding.dtype}, "
        f"L2 norm={np.linalg.norm(result.embedding):.6f}"
    )
    print(f"Embedding NPY: {result.output_path}")
    print(f"Metadata JSON: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
