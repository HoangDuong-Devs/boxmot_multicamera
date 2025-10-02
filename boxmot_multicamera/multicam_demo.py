"""Simple multi-camera demo using two Warehouse_012 videos (Camera_00 & Camera_02).

This pipeline shares a global identity manager and visualises both camera feeds
side-by-side along with a top-view map overlay that shows the re-identified
tracks as coloured dots.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.multicam import GlobalIDManager
from boxmot.qdrant.qdrant_long_reid import _HAVE_QDRANT
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS
from boxmot.utils.homography_mapper import HomographyProjector
from boxmot.trackers.botsort.basetrack import TrackState

# Add YOLOv7 repo to path
YOLOV7_ROOT = Path(__file__).parent / "yolov7"
import sys

sys.path.insert(0, str(YOLOV7_ROOT))
from yolov7.utils.datasets import letterbox  # type: ignore
from yolov7.models.experimental import attempt_load  # type: ignore
from yolov7.utils.general import non_max_suppression, scale_coords  # type: ignore


@dataclass
class CameraConfig:
    name: str
    source: Path
    tracker_camera_id: str
    calibration_camera: str


@dataclass
class DemoConfig:
    weights: Path
    tracker_type: str = "botsort"
    detect_class: int = 2
    conf_thres: float = 0.5
    iou_thres: float = 0.6
    device: str = "cuda:0"
    out_path: Path = Path("mc_demo_1.avi")
    img_size: int = 640
    cam_display_width: int = 720
    map_panel_height: int = 540
    map_image: Path = Path(
        "homography-computation/PhysicalAI-SmartSpaces_py/MTMC_Tracking_2025/train/Warehouse_012/map.png"
    )
    calibration_json: Path = Path(
        "homography-computation/PhysicalAI-SmartSpaces_py/MTMC_Tracking_2025/train/Warehouse_012/calibration.json"
    )
    map_rotation_deg: float = 180.0
    map_flip_x: bool = True
    map_flip_y: bool = False
    # NEW: khoảng hở giữa cột video và map (đẩy map sang phải)
    layout_gap_px: int = 24
    cam_vertical_gap_px: int = 8 # khoảng hở giữa 2 video (trái)
    map_width_ratio_in_col: float = 0.75  # map = 3/4 bề ngang cột phải

COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (192, 192, 192),
    (128, 128, 128),
    (64, 0, 0),
    (0, 64, 0),
    (0, 0, 64),
    (255, 165, 0),
    (255, 105, 180),
    (173, 216, 230),
]

VIZ_SCALE = 1.25

def _scaled(val: float, minv: int = 1) -> int:
    return max(minv, int(round(val * VIZ_SCALE)))

def put_text(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    fs = 0.6 * VIZ_SCALE                     # trước đây: 0.6
    th = _scaled(1)                           # trước đây: 1
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)

def visualize_tracks(
    frame: np.ndarray,
    outputs: np.ndarray,
    logs: Sequence[Dict],
    pending_tracks,
    gid_lookup: Dict[int, int],
    sim_lookup: Dict[int, float],
    camera_label: str,
) -> np.ndarray:
    vis = frame.copy()
    logs_map = {log["track_id"]: log for log in logs}

    for det in outputs:
        x1, y1, x2, y2, tid, score, *_ = det
        x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
        color = COLORS[tid % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        gid = gid_lookup.get(tid)
        sim = sim_lookup.get(tid)
        prefix = f"ID:{tid}"
        if gid is not None:
            prefix += f" (G{gid})"
        log = logs_map.get(tid)
        if log:
            text = (
                f"{prefix} sr:{log.get('reid_cost', float('nan')):.2f} "
                f"lr:{log.get('long_reid_cost', float('nan')):.2f} "
                f"iou:{log.get('iou_cost', float('nan')):.2f} d:{log.get('final_cost', float('nan')):.2f}"
            )
        else:
            text = f"{prefix} ({float(score):.2f})"
        if sim is not None and np.isfinite(sim):
            text += f" gd:{1.0 - float(sim):.2f}"
        elif sim is not None:
            text += " gd:--"
        put_text(vis, text, (x1, max(20, y1 - 6)), color)

    if pending_tracks:
        for idx, p in enumerate(pending_tracks):
            x1, y1, x2, y2 = map(int, p.xyxy)
            pseudo = p.id if p.id != -1 else (10000 + idx)
            color = COLORS[pseudo % len(COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            put_text(vis, "PENDING", (x1, max(20, y1 - 6)), color)

    # Camera label to + nền mờ
    label = camera_label
    fs = 1.2        # font scale lớn hơn
    th = 2          # độ dày
    (xw, xh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x0, y0 = 10, 10
    x1, y1 = x0 + xw + 10, y0 + xh + base + 10
    # Draw a translucent background for the camera label but use single-color text
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
    # Draw label text in white without a thick black stroke
    cv2.putText(vis, label, (x0 + 5, y0 + xh), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

    return vis

def draw_on_map(
    canvas: np.ndarray,
    projector: Optional[HomographyProjector],
    boxes: Sequence[np.ndarray],
    local_ids: Sequence[int],
    global_ids: Sequence[Optional[int]],
    similarities: Sequence[Optional[float]],
    camera_label: str,
    rot90cw: bool = False,   # <--- thêm cờ xoay
) -> None:
    if projector is None or not boxes:
        return

    pts = projector.project_bboxes(boxes)
    pts = np.asarray(pts, dtype=np.float32)
    h, w = canvas.shape[:2]

    # Nếu canvas đã xoay 90° CW, đổi hệ toạ độ điểm cho khớp canvas
    if rot90cw:
        # (x, y) -> (w-1 - y, x)  (w,h là kích thước CANVAS SAU XOAY)
        pts = np.stack([w - 1 - pts[:, 1], pts[:, 0]], axis=1)

    for (x, y), lid, gid, sim in zip(pts, local_ids, global_ids, similarities):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            xi = min(max(xi, 0), w - 1)
            yi = min(max(yi, 0), h - 1)
        color = COLORS[(gid if gid is not None else lid) % len(COLORS)]
        cv2.circle(canvas, (xi, yi), 6, color, -1, lineType=cv2.LINE_AA)
        if gid is not None:
            if sim is not None and np.isfinite(sim):
                dist = 1.0 - float(sim)
                label = f"{camera_label} L{lid}->G{gid} (d={dist:.2f})"
            else:
                label = f"{camera_label} L{lid}->G{gid}"
        else:
            label = f"{camera_label} L{lid}"
        put_text(canvas, label, (xi + 8, yi - 8), color)


class CameraRuntime:
    def __init__(
        self,
        cfg: CameraConfig,
        projector: Optional[HomographyProjector],
        tracker_type: str,
        tracker_config_path: Path,
        yolo_device: torch.device,
        reid_weights: Path,
        detect_class: int,
        global_manager: GlobalIDManager,
        img_size: int,
        conf_thres: float,
        iou_thres: float,
        model,
        reid_model=None,
    ) -> None:
        self.cfg = cfg
        self.projector = projector
        self.detect_class = detect_class
        self.global_manager = global_manager
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = model
        self.device = yolo_device
        self.reid_model = reid_model

        self.cap = cv2.VideoCapture(str(cfg.source))
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {cfg.source}")

        self.tracker = create_tracker(
            tracker_type=tracker_type,
            tracker_config=tracker_config_path,
            half=False,
            per_class=False,
            reid_weights=reid_weights,
            device=yolo_device,
            camera_id=cfg.tracker_camera_id,
            reid_model=reid_model,
        )

    def release(self) -> None:
        self.cap.release()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[int], List[Optional[int]], Dict[int, int], List]:
        img, ratio, pad = letterbox(frame, new_shape=self.img_size, auto=False)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor, augment=False)[0]
            det = non_max_suppression(
                pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres
            )[0]

        detections = np.empty((0, 6), dtype=np.float32)
        if det is not None and len(det):
            det[:, :4] = scale_coords(
                img_tensor.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio, pad)
            ).round()
            people = det[det[:, 5] == self.detect_class]
            if len(people):
                detections = people.cpu().numpy().astype(np.float32, copy=False)

        outputs, logs, _, pending_tracks = self.tracker.update(detections, frame)

        gid_lookup: Dict[int, int] = {}
        map_boxes: List[np.ndarray] = []
        map_local_ids: List[int] = []
        map_global_ids: List[Optional[int]] = []
        for trk in getattr(self.tracker, "active_tracks", []):
            if not getattr(trk, "is_activated", False):
                continue
            if getattr(trk, "state", TrackState.Tracked) != TrackState.Tracked:
                continue
            tid = getattr(trk, "id", None)
            xyxy = getattr(trk, "xyxy", None)
            if tid is None or xyxy is None:
                continue
            gid = self.global_manager.assign(
                self.cfg.tracker_camera_id,
                int(tid),
                feature=getattr(trk, "curr_feat", None),
                timestamp=frame_idx,
            )
            gid_lookup[int(tid)] = gid
            map_boxes.append(np.asarray(xyxy, dtype=np.float32))
            map_local_ids.append(int(tid))
            map_global_ids.append(gid)

        sim_lookup: Dict[int, float] = {}
        map_similarities: List[Optional[float]] = []
        for tid in map_local_ids:
            sim = self.global_manager.get_similarity(self.cfg.tracker_camera_id, tid)
            if sim is not None:
                sim_lookup[tid] = sim
            map_similarities.append(sim)

        annotated = visualize_tracks(
            frame,
            outputs,
            logs,
            pending_tracks,
            gid_lookup,
            sim_lookup,
            self.cfg.name,
        )
        return (
            annotated,
            map_boxes,
            map_local_ids,
            map_global_ids,
            map_similarities,
            gid_lookup,
            list(getattr(self.tracker, "removed_stracks", [])),
        )

def clear_qdrant_collection(name: str) -> None:
    if not _HAVE_QDRANT:
        return
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host="localhost", port=6333)
        client.delete_collection(name)
        print(f"[INFO] Cleared Qdrant collection '{name}'")
    except Exception as exc:  # pragma: no cover - best effort clean-up
        print(f"[WARN] Could not clear Qdrant collection '{name}': {exc}")

def resize_with_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    ratio = target_width / float(w)
    target_height = max(1, int(round(h * ratio)))
    return cv2.resize(frame, (target_width, target_height))

def main() -> None:
    base_dir = Path(
        "homography-computation/PhysicalAI-SmartSpaces_py/MTMC_Tracking_2025/train/Warehouse_012"
    )
    camera_configs = [
        CameraConfig(
            name="Camera 00",
            source=base_dir / "videos" / "Camera_00.mp4",
            tracker_camera_id="cam00",
            calibration_camera="Camera_00",
        ),
        CameraConfig(
            name="Camera 02",
            source=base_dir / "videos" / "Camera_02.mp4",
            tracker_camera_id="cam02",
            calibration_camera="Camera_02",
        ),
    ]

    demo_cfg = DemoConfig(weights=Path("best.pt"))

    # Ensure trackers share the same run UID
    os.environ.setdefault("BOXMOT_RUN_UID", "warehouse_demo")

    clear_qdrant_collection("long_term_reid")

    device = torch.device(demo_cfg.device)
    model = attempt_load(str(demo_cfg.weights), map_location="cpu")
    model.to(device)
    model.eval()

    global_manager = GlobalIDManager()

    projectors: Dict[str, Optional[HomographyProjector]] = {}
    for cfg in camera_configs:
        try:
            projector = HomographyProjector.from_sources(
                map_image=demo_cfg.map_image,
                calibration_path=demo_cfg.calibration_json,
                calibration_camera=cfg.calibration_camera,
                rotation_deg=demo_cfg.map_rotation_deg,
                flip_x=demo_cfg.map_flip_x,
                flip_y=demo_cfg.map_flip_y,
            )
        except Exception as exc:
            print(f"[WARN] Homography init failed for {cfg.name}: {exc}")
            projector = None
        projectors[cfg.tracker_camera_id] = projector

    tracker_config_path = TRACKER_CONFIGS / f"{demo_cfg.tracker_type}.yaml"
    reid_weights = Path("osnet_x1_0_msmt17.pt")

    reid_backend = None
    if demo_cfg.tracker_type == "botsort":
        try:
            reid_backend = ReidAutoBackend(
                weights=reid_weights,
                device=device,
                half=False,
            ).model
        except Exception as exc:
            print(f"[WARN] Failed to initialize shared ReID backend: {exc}")
            reid_backend = None

    cameras = [
        CameraRuntime(
            cfg=cfg,
            projector=projectors[cfg.tracker_camera_id],
            tracker_type=demo_cfg.tracker_type,
            tracker_config_path=tracker_config_path,
            yolo_device=device,
            reid_weights=reid_weights,
            detect_class=demo_cfg.detect_class,
            global_manager=global_manager,
            img_size=demo_cfg.img_size,
            conf_thres=demo_cfg.conf_thres,
            iou_thres=demo_cfg.iou_thres,
            model=model,
            reid_model=reid_backend,
        )
        for cfg in camera_configs
    ]

    base_map = None
    for proj in projectors.values():
        if proj is not None:
            base_map = proj.base_map.copy()
            break
    if base_map is None:
        base_map = np.zeros((600, 800, 3), dtype=np.uint8)

    writer = None
    frame_idx = 0
    start_time = time.time()
    try:
        while True:
            frames_data = []
            for cam in cameras:
                ok, frame = cam.read_frame()
                if not ok:
                    frames_data = []
                    break
                (
                    annotated,
                    boxes,
                    lids,
                    gids,
                    similarities,
                    gid_lookup,
                    removed,
                ) = cam.process_frame(
                    frame, frame_idx
                )
                frames_data.append(
                    {
                        "camera": cam,
                        "annotated": annotated,
                        "boxes": boxes,
                        "local_ids": lids,
                        "global_ids": gids,
                        "similarities": similarities,
                        "gid_lookup": gid_lookup,
                        "removed": removed,
                    }
                )
                for rem in removed:
                    rid = getattr(rem, "id", None)
                    if rid is not None:
                        global_manager.mark_lost(cam.cfg.tracker_camera_id, int(rid))
                if removed:
                    try:
                        cam.tracker.removed_stracks.clear()
                    except Exception:
                        pass

            if len(frames_data) != len(cameras):
                break

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"[INFO] Processed {frame_idx} frames")
            map_canvas = base_map.copy()
            map_canvas = cv2.rotate(base_map.copy(), cv2.ROTATE_90_CLOCKWISE)

            # vẽ điểm của tất cả camera lên cùng một canvas đã xoay
            for item in frames_data:
                draw_on_map(
                    map_canvas,
                    item["camera"].projector,
                    item["boxes"],
                    item["local_ids"],
                    item["global_ids"],
                    item["similarities"],
                    item["camera"].cfg.name,
                    rot90cw=True,   # <-- đúng tên tham số
                )

            # 1) Cột trái (video) – chiều ngang cố định = col_width
            col_width = int(demo_cfg.cam_display_width)
            cam_frames = [resize_with_width(item["annotated"], col_width) for item in frames_data]
            left_col_h = sum(f.shape[0] for f in cam_frames) + (len(cam_frames) - 1) * demo_cfg.cam_vertical_gap_px

            # 2) Cột phải (map) – WIDTH = 80% cột; dư chiều dọc thì CROP đều 2 đầu
            map_h0, map_w0 = map_canvas.shape[:2]
            side_margin_ratio = 0.10                             # 10% mỗi bên
            left_margin_px = int(round(col_width * side_margin_ratio))

            if map_w0 <= 0 or map_h0 <= 0:
                map_panel = np.zeros((1, 1, 3), dtype=np.uint8)
            else:
                target_map_w = max(1, col_width - 2 * left_margin_px)  # = 80% col_width
                scale = target_map_w / float(map_w0)                   # scale theo WIDTH
                scaled_h = max(1, int(round(map_h0 * scale)))
                map_scaled = cv2.resize(map_canvas, (target_map_w, scaled_h), interpolation=cv2.INTER_CUBIC)

                # Crop dọc nếu cao hơn cột trái
                if map_scaled.shape[0] > left_col_h:
                    start = (map_scaled.shape[0] - left_col_h) // 2
                    map_panel = map_scaled[start:start + left_col_h, :, :]
                else:
                    map_panel = map_scaled

            # 3) Khung cuối: 2 cột bằng nhau (chiều cao = cột trái)
            final_width  = 2 * col_width
            final_height = left_col_h
            final_frame  = np.zeros((final_height, final_width, 3), dtype=np.uint8)

            # 4) Vẽ cột trái (2 video xếp dọc)
            y = (final_height - left_col_h) // 2
            for i, frame_vis in enumerate(cam_frames):
                h, w = frame_vis.shape[:2]
                x = (col_width - w) // 2
                final_frame[y:y+h, x:x+w] = frame_vis
                y += h
                if i < len(cam_frames) - 1:
                    y += demo_cfg.cam_vertical_gap_px

            # 5) Vẽ cột phải (map) – đặt lệch trái 10% cột, chừa 10% bên phải
            x_map = col_width + left_margin_px                   # lệch 10% bên trái
            y_map = (final_height - map_panel.shape[0]) // 2     # đã crop nếu cần
            final_frame[y_map:y_map + map_panel.shape[0], x_map:x_map + map_panel.shape[1]] = map_panel

            put_text(final_frame, f"Frame {frame_idx}", (15, final_height - 30), (255, 255, 255))
            elapsed = time.time() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            put_text(final_frame, f"FPS: {fps:.2f}", (15, final_height - 60), (255, 255, 255))

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(
                    str(demo_cfg.out_path), fourcc, 25, (final_width, final_height)
                )
            writer.write(final_frame)
    finally:
        if writer is not None:
            writer.release()
        for cam in cameras:
            cam.release()
        print(f"[INFO] Demo finished. Video saved to {demo_cfg.out_path}")

if __name__ == "__main__":
    main()
