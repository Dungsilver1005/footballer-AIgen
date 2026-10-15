"""Stage 3 public API. Importing never loads models, runs inference or writes files."""

from .embedder import ArcFaceEmbedder
from .processor import EmbeddingResult, load_stage1_crop, process_aligned_face

__all__ = ["ArcFaceEmbedder", "EmbeddingResult", "load_stage1_crop", "process_aligned_face"]
