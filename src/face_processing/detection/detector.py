"""CPU-only SCRFD adapter using an explicitly supplied local ONNX file.

Dependencies: NumPy, onnxruntime and InsightFace (with its ONNX/OpenCV
dependencies). Nothing is installed, downloaded or loaded on import.
The caller must supply SCRFD-2.5G-KPS weights; output shape alone cannot
verify the model's identity or computational size.

Backend contract: InsightFace's SCRFD(session=...), prepare(ctx_id=-1),
and detect(..., max_num=0), as defined in its model_zoo/scrfd.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from insightface.model_zoo.scrfd import SCRFD


LANDMARK_ORDER = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceDetection:
    """One face in pixels of the supplied, unresized image.

    bbox is (x1, y1, x2, y2); landmarks retain SCRFD's five-point order:
    image-left eye, image-right eye, nose, image-left mouth, image-right mouth.
    Coordinates stay as floats for future alignment; no clipping or alignment
    is applied to this record. confidence is in [0, 1].
    """

    bbox: tuple[float, float, float, float]
    landmarks: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    confidence: float


class SCRFDDetector:
    """Configure without side effects; call load_model() before detect()."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        if len(input_size) != 2 or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value % 32 != 0
            for value in input_size
        ):
            raise ValueError("Input width and height must be positive multiples of 32")
        for name, value in (
            ("confidence_threshold", confidence_threshold),
            ("nms_threshold", nms_threshold),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._model: SCRFD | None = None

    def load_model(self) -> None:
        """Load local SCRFD KPS weights through a CPU-only ONNX session.

        Missing weights/dependencies raise clear errors. This method never
        invokes FaceAnalysis, a model router or any download helper.
        Static model dimensions must match the configured input_size.
        """
        if self._model is not None:
            logger.debug("SCRFD model is already loaded: %s", self.model_path)
            return
        try:
            exists = self.model_path.is_file()
            logger.info("SCRFD model: %s (exists=%s)", self.model_path.resolve(), exists)
            if not exists:
                raise FileNotFoundError(
                    f"Local SCRFD-2.5G-KPS weights missing: {self.model_path}"
                )
            if self.model_path.suffix.lower() != ".onnx":
                raise ValueError("SCRFD weights must be a local .onnx file")
            try:
                import onnxruntime as ort
                from insightface.model_zoo.scrfd import SCRFD
            except ImportError as exc:
                raise ImportError(
                    "Face detection requires onnxruntime (CPU) and insightface, "
                    "including its onnx/OpenCV dependencies. Prepare them manually. "
                    f"Import failed: {exc}"
                ) from exc

            providers = ["CPUExecutionProvider"]
            available_providers = ort.get_available_providers()
            logger.info("ONNX Runtime available providers: %s", available_providers)
            logger.info("ONNX Runtime requested providers: %s", providers)
            if providers[0] not in available_providers:
                raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            logger.debug(
                "ONNX inputs: %s",
                [(item.name, item.shape, item.type) for item in inputs],
            )
            logger.debug(
                "ONNX outputs (%d): %s",
                len(outputs), [(item.name, item.shape, item.type) for item in outputs],
            )
            if len(inputs) != 1 or len(inputs[0].shape) != 4:
                raise ValueError("Expected a single NCHW SCRFD input")
            if inputs[0].type != "tensor(float)" or inputs[0].shape[1] != 3:
                raise ValueError("Expected a float32 SCRFD input with three color channels")
            shape = inputs[0].shape
            if all(isinstance(value, int) and value > 0 for value in shape[2:4]):
                fixed_size = (shape[3], shape[2])
                if fixed_size != self.input_size:
                    raise ValueError(f"Model requires input_size={fixed_size}")
            if len(outputs) != 9:
                raise ValueError("Expected the nine-output SCRFD KPS ONNX export")

            # Inject the session so SCRFD cannot create one with default providers.
            model = SCRFD(session=session)
            if not model.use_kps:
                raise ValueError("SCRFD model does not provide facial landmarks")
            model.prepare(
                ctx_id=-1,
                input_size=self.input_size,
                det_thresh=self.confidence_threshold,
                nms_thresh=self.nms_threshold,
            )
            session.disable_fallback()
            active_providers = session.get_providers()
            logger.info("ONNX Runtime active providers: %s", active_providers)
            if active_providers != providers:
                raise RuntimeError("SCRFD must use CPUExecutionProvider exclusively")
        except Exception as exc:
            logger.error(
                "MODEL_LOAD_ERROR: %s: %s", self.model_path, exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            raise
        self._model = model
        logger.info(
            "SCRFD model loaded successfully (input_size=%s, five landmarks)",
            self.input_size,
        )

    def detect(self, image: NDArray[np.uint8]) -> list[FaceDetection]:
        """Detect all faces in an HxWx3 uint8 BGR image; do not select one."""
        if self._model is None:
            raise RuntimeError("SCRFD model is not loaded; call load_model() first")
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("Expected a non-empty HxWx3 uint8 BGR image")

        boxes, landmarks = self._model.detect(
            np.ascontiguousarray(image), input_size=self.input_size, max_num=0
        )
        boxes = np.asarray(boxes)
        if boxes.ndim != 2 or boxes.shape[1] != 5:
            raise ValueError("SCRFD returned invalid bbox/confidence data")
        if len(boxes) == 0:
            return []
        landmarks = np.asarray(landmarks)
        if landmarks.shape != (len(boxes), 5, 2):
            raise ValueError("SCRFD must return five landmarks for every face")
        if not np.isfinite(boxes).all() or not np.isfinite(landmarks).all():
            raise ValueError("SCRFD returned non-finite detection coordinates")
        if ((boxes[:, 4] < 0) | (boxes[:, 4] > 1)).any():
            raise ValueError("SCRFD confidence must be in [0, 1]")

        results = []
        for box, points in zip(boxes, landmarks):
            x1, y1, x2, y2, confidence = (float(value) for value in box)
            point_pairs = tuple((float(x), float(y)) for x, y in points)
            results.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    landmarks=(
                        point_pairs[0], point_pairs[1], point_pairs[2],
                        point_pairs[3], point_pairs[4],
                    ),
                    confidence=confidence,
                )
            )
        return results
