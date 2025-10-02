# yolov7_track_pending.py

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
import time

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS
from boxmot.multicam import GlobalIDManager

# Add YOLOv7 repo to path so its utils can be imported
YOLOV7_ROOT = Path(__file__).parent / "yolov7"
sys.path.insert(0, str(YOLOV7_ROOT))

from yolov7.utils.datasets      import letterbox
from yolov7.models.experimental import attempt_load
from yolov7.utils.general       import non_max_suppression, scale_coords

def clear_qdrant_collection(name="long_term_reid"):
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.delete_collection(name)
        print(f"[INFO] Qdrant collection '{name}' deleted successfully.")
    except Exception as e:
        print(f"[WARN] could not delete collection '{name}': {e}")

@dataclass
class TrackingConfig:
    source      : str   = r"/home/duonghn/boxmot_rebuild1/test_video/aicity.mp4"
    weights     : str   = "best.pt"
    detect_class: int   = 2
    tracker_type: str   = "botsort"
    device      : str   = "0"
    conf_thres  : float = 0.55
    iou_thres   : float = 0.7
    output      : str   = "aicity3.avi"
    img_size    : int   = 640
    half        : bool  = False
    per_class   : bool  = False
    camera_id   : str   = "cam0"

def put_text_styled(img, text, org, color, scale=0.6, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def put_multiline(img, lines, x, y, color, line_h=16, scale=0.55, thick=1):
    for i, s in enumerate(lines):
        put_text_styled(img, s, (x, y + i * line_h), color, scale, thick)

def safe_fmt(v, fmt=".2f"):
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return "nan"
        return format(f, fmt)
    except Exception:
        return "nan"

def visualize_with_logs(frame, outputs, logs, colors, pending_tracks=None, global_ids=None):
    # Vẽ các track chính
    gid_lookup = global_ids or {}
    for det in outputs:
        x1, y1, x2, y2, tid, score, *_ = det
        x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
        color = colors[tid % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        log = next((log for log in logs if log["track_id"] == tid), None)

        if log:
            gid = gid_lookup.get(tid)
            id_text = f"ID:{tid}"
            if gid is not None:
                id_text += f" (G{gid})"
            text = (
                f"{id_text} sr:{log.get('reid_cost', float('nan')):.2f} lr:{log.get('long_reid_cost', float('nan')):.2f} "
                f"iou:{log.get('iou_cost', float('nan')):.2f} "
                f"d:{log.get('final_cost', float('nan')):.2f}"
            )
        else:
            gid = gid_lookup.get(tid)
            base = f"ID:{tid}"
            if gid is not None:
                base += f" (G{gid})"
            text = f"{base}({score:.2f})"

        put_text_styled(frame, text, (x1, max(15, y1 - 5)), color, scale=0.55, thickness=1)

    # Vẽ các pending tracks
    if pending_tracks:
        for idx, p in enumerate(pending_tracks):
            x1, y1, x2, y2 = map(int, p.xyxy)
            pending_id     = p.id if p.id != -1 else (10000 + idx)  # tạo pseudo-id để chọn màu
            color          = colors[pending_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            put_text_styled(frame, "PENDING", (x1, max(15, y1 - 5)), color, scale=0.55, thickness=1)

    return frame

def visualize_full_hd(video_frame, tracks, rows=6, cols=5, padding=5, text_height=20):
    target_w, target_h = 1920, 1080
    grid_h = target_h
    cell_h = ((grid_h - (rows + 1) * padding - rows * text_height) // rows)
    cell_w = int(cell_h * (2 / 3))

    grid_w  = cols * (cell_w + padding) + padding
    video_w = target_w - grid_w
    video_frame_resized = cv2.resize(video_frame, (video_w, target_h))

    grid_panel  = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240
    total_slots = rows * cols

    for idx in range(total_slots):
        row = idx // cols
        col = idx % cols
        x_start = padding + col * (cell_w + padding)
        y_start = padding + row * (cell_h + text_height + padding)

        if idx < len(tracks):
            track = tracks[idx]
            img = track.get("image", None)
            if img is None or img.size == 0:
                # Ô trống an toàn nếu thiếu ảnh
                cv2.rectangle(grid_panel, (x_start, y_start),
                              (x_start + cell_w, y_start + cell_h),
                              (200, 200, 200), -1)
            else:
                img_crop = cv2.resize(img, (cell_w, cell_h))
                grid_panel[y_start:y_start + cell_h, x_start:x_start + cell_w] = img_crop

                is_lost = bool(track.get("is_lost", False))
                if is_lost:
                    center_x = x_start + cell_w // 2
                    center_y = y_start + cell_h // 2
                    L_scale = max(1.0, min(3.0, cell_h / 120.0))
                    L_thick = max(2, int(cell_h / 40))
                    text = "L"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, L_scale, L_thick)
                    cv2.putText(
                        grid_panel, text,
                        (center_x - tw // 2, center_y + th // 4),
                        cv2.FONT_HERSHEY_SIMPLEX, L_scale, (0, 0, 255), L_thick, cv2.LINE_AA
                    )

                suffix = ""
                lr_cost = track.get("long_reid_cost", None)
                if lr_cost is not None:
                    try:
                        f = float(lr_cost)
                        if not (np.isnan(f) or np.isinf(f)):
                            suffix += f" {f:.2f}"
                    except Exception:
                        pass

                label = f"ID: {track.get('id', '?')}{suffix}"
                put_text_styled(
                    grid_panel, label,
                    (x_start + 3, y_start + cell_h + 15),
                    (0, 0, 0), scale=0.55, thickness=1
                )
        else:
            cv2.rectangle(grid_panel, (x_start, y_start),
                          (x_start + cell_w, y_start + cell_h),
                          (200, 200, 200), -1)

    final_display = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    final_display[:, :video_w] = video_frame_resized
    final_display[:, video_w:] = grid_panel
    return final_display

def run_tracking(cfg: TrackingConfig) -> None:
    selected_device = torch.device("cuda:0")
    clear_qdrant_collection("long_term_reid")
    print(f"[INFO] Using device: {selected_device}")

    model = attempt_load(cfg.weights, map_location="cpu")
    model.to(selected_device)
    model.eval()
    print(f"[INFO] YOLOv7 model on: {next(model.parameters()).device}")

    tracker = create_tracker(
        tracker_type  =cfg.tracker_type,
        tracker_config=TRACKER_CONFIGS / f"{cfg.tracker_type}.yaml",
        half          =cfg.half,
        per_class     =cfg.per_class,
        reid_weights  =Path("osnet_x1_0_msmt17.pt"),
        device        =selected_device,
        camera_id     =cfg.camera_id,
    )

    global_id_manager = GlobalIDManager(
        cost_threshold=0.1,            # Cost threshold (0.0-1.0, lower = more strict)
        cost_margin=0.05,              # Cost margin for ambiguous matches
        historical_cost_threshold=0.4, # Historical cost threshold (looser than current)
        # Extended timeouts
        lost_timeout_seconds=600,      # Tăng từ 300s (5min) lên 600s (10min)
        inactive_timeout_seconds=7200, # Tăng từ 3600s (1h) lên 7200s (2h)
        # Enable Qdrant for historical evidence
        use_qdrant=True,
        qdrant_collection="real_tracking_session"
    )

    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {cfg.source}")

    # --- NEW: In ra FPS video gốc & fallback nếu thiếu ---
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if raw_fps and not np.isnan(raw_fps) and raw_fps > 0:
        fps = float(raw_fps)
        print(f"[INFO] Source FPS (gốc): {fps:.2f}")
    else:
        fps = 30.0
        print(f"[WARN] Video không có/không báo FPS hợp lệ (got={raw_fps}). "
            f"Dùng mặc định: {fps:.2f}")

    out = cv2.VideoWriter(cfg.output, cv2.VideoWriter_fourcc(*"XVID"), fps, (1920, 1080))

    colors: List[tuple] = [
        (255, 0  , 0  ), (0  , 255, 0  ), (0  , 0  , 255),
        (255, 255, 0  ), (255, 0  , 255), (0  , 255, 255),
        (128, 0  , 0  ), (0  , 128, 0  ), (0  , 0  , 128),
        (128, 128, 0  ), (128, 0  , 128), (0  , 128, 128),
        (192, 192, 192), (128, 128, 128), (64 , 0  , 0  ),
        (0  , 64 , 0  ), (0  , 0  , 64 ), (255, 165, 0  ),
        (255, 105, 180), (173, 216, 230),
    ]

    # === FPS & Frame Index
    frame_idx = 0
    prev_time = time.time()
    start_time = prev_time  # tổng thời gian chạy

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        print(f"Frame: {frame_idx}")
        curr_time = time.time()
        fps_real = 1.0 / (curr_time - prev_time)
        avg_fps = frame_idx / (curr_time - start_time)  # FPS trung bình tính tại thời điểm này
        prev_time = curr_time

        img, ratio, pad = letterbox(frame, new_shape=cfg.img_size, auto=False)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(selected_device).float() / 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]
            det = non_max_suppression(pred, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres)[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio, pad)).round()
            det = det[det[:, 5] == 2]  # Filter class (person)
            if len(det):
                outputs, logs, tracks_for_visual, pending_tracks = tracker.update(det.cpu().numpy(), frame)
                
                # Collect existing track IDs from tracker (active + lost)
                active_tracks = getattr(tracker, "active_tracks", [])
                lost_tracks = getattr(tracker, "lost_stracks", [])
                
                existing_track_ids = set()
                for trk in active_tracks:
                    tid = getattr(trk, "id", None)
                    if tid is not None:
                        existing_track_ids.add(int(tid))
                for trk in lost_tracks:
                    tid = getattr(trk, "id", None)
                    if tid is not None:
                        existing_track_ids.add(int(tid))

                global_id_map: dict[int, int] = {}
                frame_stamp = getattr(tracker, "frame_count", frame_idx)
                
                for trk in active_tracks:
                    if not getattr(trk, "is_activated", False):
                        continue
                    tid = getattr(trk, "id", None)
                    xy = getattr(trk, "xyxy", None)
                    if tid is None or xy is None:
                        continue
                    tid_int = int(tid)
                    
                    # Get current feature for assignment
                    feature = getattr(trk, "curr_feat", None)
                    
                    # Pass existing_track_ids so GlobalIDManager can detect removed tracks
                    gid = global_id_manager.assign(
                        cfg.camera_id,
                        tid_int,
                        feature=feature,
                        timestamp=frame_stamp,
                        existing_track_ids=existing_track_ids,  # NEW: sync tracker state
                    )
                    
                    # Debug logging for reuse detection
                    assignment_info = global_id_manager.get_assignment_info(cfg.camera_id, tid_int)
                    if assignment_info:
                        match_type = assignment_info.get('match_type', 'unknown')
                        similarity = assignment_info.get('similarity', 0.0)
                        cost = 1.0 - similarity  # Convert to cost for display
                        
                        # Only log important events (not existing_assignment)
                        if match_type in ['lost_reactivation', 'inactive_reactivation']:
                            print(f"🔄 GLOBAL ID REUSE: Track {tid_int} -> Global {gid}")
                            print(f"   Match type: {match_type}, Cost: {cost:.3f} (sim: {similarity:.3f})")
                        elif match_type in ['cross_camera_match', 'historical_match']:
                            print(f"� CROSS-CAMERA MATCH: Track {tid_int} -> Global {gid}")
                            print(f"   Match type: {match_type}, Cost: {cost:.3f} (sim: {similarity:.3f})")
                        elif match_type == 'new_entity' and gid != tid_int:
                            print(f"🆕 NEW GLOBAL ID: Track {tid_int} -> Global {gid}")
                        # Skip logging for 'existing_assignment' (too verbose)
                    
                    global_id_map[tid_int] = gid

                logs_dict = {log["track_id"]: log for log in logs}
                full_logs = []
                for d in outputs:
                    tid = int(d[4])
                    score = float(d[5])
                    log = logs_dict.get(tid, {})
                    base_log = {
                        "track_id"      : tid,
                        "conf"          : score,
                        "reid_cost"     : log.get("reid_cost", float("nan")),
                        "long_reid_cost": log.get("long_reid_cost", float("nan")),
                        "iou_cost"      : log.get("iou_cost", float("nan")),
                        "final_cost"    : log.get("final_cost", float("nan")),
                    }
                    full_logs.append(base_log)

                frame_with_logs = visualize_with_logs(
                    frame.copy(),
                    outputs,
                    full_logs,
                    colors,
                    pending_tracks=pending_tracks,
                    global_ids=global_id_map,
                )

                active_ids = {int(d[4]) for d in outputs} 
                for t in tracks_for_visual:
                    t_id = int(t["id"])
                    t["is_lost"] = t_id not in active_ids
                    t["long_reid_cost"] = logs_dict.get(t_id, {}).get("long_reid_cost", float("nan"))
                tracks_for_visual.sort(key=lambda t: t["id"])

                display_frame = visualize_full_hd(frame_with_logs, tracks_for_visual)
            else:
                display_frame = visualize_full_hd(frame, [])
        else:
            display_frame = visualize_full_hd(frame, [])

        # === Hiển thị FPS hiện tại & FPS trung bình
        info_text = f"Frame: {frame_idx} | FPS: {fps_real:.2f}"
        avg_text = f"Avg FPS: {avg_fps:.2f}"
        put_text_styled(display_frame, info_text, (20, 40), (0, 255, 0)  , scale=0.9, thickness=2)
        put_text_styled(display_frame, avg_text,  (20, 80), (0, 255, 255), scale=0.9, thickness=2)

        out.write(display_frame)

    cap.release()
    out.release()

    # In ra FPS trung bình sau khi xử lý toàn bộ video
    total_time = time.time() - start_time
    avg_fps_final = frame_idx / total_time if total_time > 0 else 0
    print(f"\n✅ Tracking finished. Results saved to {cfg.output}")
    print(f"📊 Average FPS (final): {avg_fps_final:.2f}")

if __name__ == "__main__":
    config = TrackingConfig()
    run_tracking(config)
