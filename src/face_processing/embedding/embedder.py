"""Explicit, CPU-only inference using a local InsightFace ArcFace ONNX model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..detection.processor import PROJECT_ROOT
from .validator import (
    EMBEDDING_DIMENSION,
    validate_aligned_face,
    validate_model_io,
    validate_normalized_embedding,
    validate_raw_embedding,
)

if TYPE_CHECKING:
    from onnxruntime import InferenceSession


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_recognition" / "w600k_r50.onnx"
logger = logging.getLogger(__name__)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative API/CLI paths using the same project root as Stage 1."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


class ArcFaceEmbedder:
    """Configure only on construction; explicitly call load_model() before use."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = resolve_project_path(model_path)
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.input_mean: float | None = None
        self.input_std: float | None = None
        self._session: InferenceSession | None = None

    def load_model(self) -> None:
        """Inspect normalization once, then load a strictly CPU-only ORT session.

        Follow InsightFace model_zoo/arcface_onnx.py: Sub/_minus AND Mul/_mul
        among the first eight node names identify graph-side normalization.
        Other exports use Python-side mean=std=127.5. Tensor shape is never
        used to select normalization. onnx is already a Stage 1 dependency.
        """
        if self._session is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Local ArcFace model does not exist: {self.model_path}")
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(f"ArcFace model must be a local .onnx file: {self.model_path}")
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ArcFace requires the existing onnx and onnxruntime (CPU) dependencies. "
                f"Prepare them manually; import failed: {exc}"
            ) from exc

        try:
            model = onnx.load(str(self.model_path), load_external_data=False)
            if not model.graph.node:
                raise ValueError("ONNX graph contains no nodes")
            names = [node.name for node in model.graph.node[:8]]
            graph_normalization = (
                any(name.startswith(("Sub", "_minus")) for name in names)
                and any(name.startswith(("Mul", "_mul")) for name in names)
            )
            # Do not retain a second copy of the model weights after inspection.
            del model
        except Exception as exc:
            raise ValueError(f"Cannot inspect ArcFace ONNX graph {self.model_path}: {exc}") from exc

        providers = ["CPUExecutionProvider"]
        if providers[0] not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
        try:
            session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as exc:
            raise ValueError(f"Cannot load ArcFace ONNX model {self.model_path}: {exc}") from exc
        disable_fallback = getattr(session, "disable_fallback", None)
        if callable(disable_fallback):
            disable_fallback()
        if session.get_providers() != providers:
            raise RuntimeError("ArcFace must use CPUExecutionProvider exclusively")
        input_name, output_name = validate_model_io(session.get_inputs(), session.get_outputs())

        self.input_name, self.output_name = input_name, output_name
        self.input_mean = 0.0 if graph_normalization else 127.5
        self.input_std = 1.0 if graph_normalization else 127.5
        self._session = session
        logger.debug("ArcFace first eight ONNX nodes: %s", names)
        logger.info(
            "ArcFace loaded: %s; CPU-only; RGB input; mean=%s std=%s; normalization=%s",
            self.model_path, self.input_mean, self.input_std,
            "onnx_graph" if graph_normalization else "python",
        )

    def preprocess(self, aligned_face_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return contiguous (1,3,112,112) float32; no resize/alignment/channel swap.

        InsightFace's swapRB=True converts its BGR image input to RGB. This
        pipeline already supplies RGB, so applying that swap here is incorrect.
        """
        validate_aligned_face(aligned_face_rgb)
        if self._session is None:
            raise RuntimeError("ArcFace model is not loaded; call load_model() first")
        pixels = aligned_face_rgb.astype(np.float32)
        pixels = (pixels - np.float32(self.input_mean)) / np.float32(self.input_std)
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, ...])

    def extract_embedding(self, aligned_face_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return a finite, L2-normalized float32 vector of shape (512,)."""
        tensor = self.preprocess(aligned_face_rgb)
        try:
            outputs = self._session.run([self.output_name], {self.input_name: tensor})
        except Exception as exc:
            raise RuntimeError(f"ArcFace inference failed for {self.model_path}: {exc}") from exc
        if (
            len(outputs) != 1
            or not isinstance(outputs[0], np.ndarray)
            or outputs[0].shape != (1, EMBEDDING_DIMENSION)
            or outputs[0].dtype != np.float32
        ):
            raise ValueError("ArcFace inference must return one float32 tensor with shape (1, 512)")
        raw = outputs[0][0]
        norm = validate_raw_embedding(raw)
        embedding = (raw.astype(np.float64) / norm).astype(np.float32)
        validate_normalized_embedding(embedding)
        return embedding
