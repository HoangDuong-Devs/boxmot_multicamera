"""Homography utilities for projecting detections onto a top-view map."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2 as cv
import numpy as np


def _normalize_h(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("Homography matrix must be 3x3")
    if abs(H[2, 2]) <= 1e-12:
        return H
    return H / H[2, 2]


def _translation_scale(scale: float, txy: dict) -> np.ndarray:
    s, tx, ty = float(scale), float(txy.get("x", 0.0)), float(txy.get("y", 0.0))
    return np.array([[s, 0.0, s * tx], [0.0, s, s * ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_center(w: float, h: float, deg: float) -> np.ndarray:
    M = cv.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = M
    return out


def _flip_center(w: float, h: float, flip_x: bool, flip_y: bool) -> np.ndarray:
    sx, sy = (-1 if flip_x else 1), (-1 if flip_y else 1)
    cx, cy = w / 2.0, h / 2.0
    T1 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    return T1 @ D @ T2


def load_calibrated_homography(
    calibration_path: Path,
    camera_id: str,
    map_size: Tuple[int, int],
    rotation_deg: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    """Builds img->map homography using a calibration JSON in SmartSpaces format."""
    calib = json.load(Path(calibration_path).open("r", encoding="utf-8"))
    sensors = calib.get("sensors", [])
    cam = next((s for s in sensors if s.get("id") == camera_id), None)
    if cam is None:
        raise KeyError(f"Camera '{camera_id}' not found in calibration file")

    H = _normalize_h(np.asarray(cam["homography"], dtype=np.float64))
    H_inv = np.linalg.inv(H)
    T = _translation_scale(cam["scaleFactor"], cam["translationToGlobalCoordinates"])
    mw, mh = float(map_size[0]), float(map_size[1])
    orient = _rotation_center(mw, mh, rotation_deg) @ _flip_center(mw, mh, flip_x, flip_y)
    return _normalize_h(orient @ T @ H_inv)


@dataclass
class HomographyProjector:
    """Projects image coordinates onto a map using a homography matrix."""

    map_image_path: Path
    H_img2map: np.ndarray

    def __post_init__(self) -> None:
        self.map_image_path = Path(self.map_image_path)
        if not self.map_image_path.exists():
            raise FileNotFoundError(f"Map image not found: {self.map_image_path}")
        self.base_map = cv.imread(str(self.map_image_path), cv.IMREAD_COLOR)
        if self.base_map is None:
            raise RuntimeError(f"Failed to load map image: {self.map_image_path}")
        self.H_img2map = _normalize_h(self.H_img2map)

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.base_map.shape[:2]
        return w, h

    def project_points(self, pts: Sequence[Tuple[float, float]]) -> np.ndarray:
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts_arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv.perspectiveTransform(pts_arr, self.H_img2map)
        return projected.reshape(-1, 2)

    def project_bboxes(self, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        centers = []
        for b in boxes:
            if b is None or len(b) < 4:
                centers.append((0.0, 0.0))
                continue
            x1, y1, x2, y2 = map(float, b[:4])
            centers.append(((x1 + x2) / 2.0, y2))  # bottom centre
        return self.project_points(centers)

    def draw_tracks(
        self,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        colors: Optional[Sequence[Tuple[int, int, int]]] = None,
        radius: int = 6,
    ) -> np.ndarray:
        canvas = self.base_map.copy()
        if len(boxes) == 0:
            return canvas
        pts = self.project_bboxes(boxes)
        h, w = canvas.shape[:2]
        for idx, (pt, label) in enumerate(zip(pts, labels)):
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (0 <= x < w and 0 <= y < h):
                continue
            color = (0, 215, 255) if colors is None else colors[idx % len(colors)]
            cv.circle(canvas, (x, y), radius, color, -1, lineType=cv.LINE_AA)
            cv.putText(
                canvas,
                label,
                (x + radius + 2, y - radius - 2),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
        return canvas

    @classmethod
    def from_sources(
        cls,
        map_image: Path,
        homography_matrix: Optional[Sequence[Sequence[float]]] = None,
        matrix_path: Optional[Path] = None,
        calibration_path: Optional[Path] = None,
        calibration_camera: Optional[str] = None,
        rotation_deg: float = 0.0,
        flip_x: bool = False,
        flip_y: bool = False,
    ) -> "HomographyProjector":
        map_image = Path(map_image)
        base = cv.imread(str(map_image), cv.IMREAD_COLOR)
        if base is None:
            raise RuntimeError(f"Failed to load map image: {map_image}")
        mh, mw = base.shape[:2]

        H = None
        if homography_matrix is not None:
            H = np.asarray(homography_matrix, dtype=np.float64)
        elif matrix_path is not None:
            matrix_path = Path(matrix_path)
            if matrix_path.suffix.lower() in {".npy", ".npz"}:
                H = np.load(str(matrix_path))
            else:
                H = np.loadtxt(str(matrix_path), dtype=np.float64)
        elif calibration_path is not None and calibration_camera is not None:
            H = load_calibrated_homography(
                Path(calibration_path),
                calibration_camera,
                map_size=(mw, mh),
                rotation_deg=rotation_deg,
                flip_x=flip_x,
                flip_y=flip_y,
            )
        else:
            raise ValueError("Homography source not provided")
        return cls(map_image, H)
