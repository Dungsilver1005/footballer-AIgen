r"""Run local SCRFD detection/cropping; relative CLI paths use project root.

From project root::

    .\.venv\Scripts\python.exe -m src.face_processing.detection --verbose

Importing this runner does not load weights or process images.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from .detector import SCRFDDetector
from .processor import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    process_directory,
)
from .validator import ProcessingStatus, validate_margin


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and crop single faces using local SCRFD weights on CPU.",
        epilog="Relative paths are resolved from project root. Existing outputs are preserved.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Local SCRFD KPS ONNX model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Input image directory (non-recursive)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for PNG crops and JSON metadata")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="Crop margin as a fraction of bbox dimensions on each side")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging and exception tracebacks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        validate_margin(args.margin)
        paths = []
        for path in (args.model, args.input, args.output):
            path = path.expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            paths.append(path.resolve())
        model_path, input_dir, output_dir = paths
        detector = SCRFDDetector(model_path)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("CONFIG_ERROR: %s", exc, exc_info=args.verbose)
        return 2

    logger.info("Crop margin: %s on each side", args.margin)
    try:
        detector.load_model()
    except Exception:
        # load_model() logs the cause and includes a traceback in verbose mode.
        return 1

    try:
        results = process_directory(detector, input_dir, output_dir, margin=args.margin)
    except OSError as exc:
        logger.error("INPUT_DIRECTORY_ERROR: %s: %s", input_dir, exc, exc_info=args.verbose)
        return 1
    except Exception as exc:
        logger.error("PROCESSING_ERROR: %s", exc, exc_info=args.verbose)
        return 1

    counts = Counter(result.status for result in results)
    print(f"Processing report: {len(results)} image(s)")
    for status in ProcessingStatus:
        print(f"  {status.name}: {counts[status]}")
    error_statuses = (
        ProcessingStatus.INVALID_IMAGE,
        ProcessingStatus.INVALID_BBOX,
        ProcessingStatus.DETECTION_ERROR,
        ProcessingStatus.OUTPUT_ERROR,
    )
    return 1 if any(counts[status] for status in error_statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
