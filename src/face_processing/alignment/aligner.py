"""Align five SCRFD landmarks to the ArcFace template without file I/O."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


# ArcFace 112x112, matching SCRFD's image-viewer left/right landmark order:
# left_eye, right_eye, nose, left_mouth, right_mouth.
_ARCFACE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def align_face(
    image: NDArray[np.uint8],
    landmarks: ArrayLike,
    output_size: tuple[int, int] = (112, 112),
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Return an aligned image and its source-to-output 2x3 matrix.

    Input must be a non-empty HxWx3 uint8 array. RGB or BGR channel order
    is preserved. Landmarks must be five finite (x, y) pairs in the same
    EXIF-normalized coordinate space as the image, in SCRFD order.
    output_size is (width, height); rectangular outputs uniformly scale
    and center the template. No encoder pixel normalization is applied.
    Invalid inputs or degenerate geometry raise ValueError.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("Expected a non-empty HxWx3 uint8 image")
    if (
        not isinstance(output_size, (tuple, list))
        or len(output_size) != 2
        or any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value <= 0
            for value in output_size
        )
    ):
        raise ValueError("Output size must contain positive integer width and height")
    width, height = (int(value) for value in output_size)

    try:
        points = np.asarray(landmarks)
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmarks must contain five numeric (x, y) pairs") from exc
    if points.shape != (5, 2) or points.dtype.kind not in "iuf":
        raise ValueError("Landmarks must contain five real numeric (x, y) pairs")
    points = points.astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError("Landmark coordinates must be finite")

    reference_scale = min(width, height) / 112.0
    reference = _ARCFACE_LANDMARKS * reference_scale
    reference += (np.array([width, height]) - 112.0 * reference_scale) / 2.0

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            source_mean = points.mean(axis=0)
            target_mean = reference.mean(axis=0)
            source = points - source_mean
            target = reference - target_mean
            if np.linalg.matrix_rank(source) < 2:
                raise ValueError("Landmarks must not be coincident or collinear")

            # Least-squares similarity: X = a*x - b*y + tx,
            # Y = b*x + a*y + ty. Its scale is hypot(a, b); no shear/reflection.
            variance = np.sum(source * source)
            a = np.sum(source * target) / variance
            b = np.sum(source[:, 0] * target[:, 1] - source[:, 1] * target[:, 0]) / variance
            linear = np.array([[a, -b], [b, a]], dtype=np.float64)
            translation = target_mean - linear @ source_mean
            transform_matrix = np.column_stack((linear, translation))
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("Cannot estimate a finite similarity transform") from exc
    if (
        not np.isfinite(transform_matrix).all()
        or np.hypot(a, b) <= np.finfo(np.float64).eps
    ):
        raise ValueError("Landmarks produce a degenerate similarity transform")

    aligned_face = cv2.warpAffine(
        np.ascontiguousarray(image),
        transform_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned_face, transform_matrix
