# botsort_utils.py

from typing import List, Tuple
import numpy as np
import cv2
from boxmot.utils.matching import iou_distance
from boxmot.trackers.botsort.botsort_track import STrack, crop_image

def joint_stracks(tlista: List["STrack"], tlistb: List["STrack"]) -> List["STrack"]:
    """
    Joins two lists of tracks, ensuring that there are no duplicates based on track IDs.

    Args:
        tlista (List[STrack]): The first list of tracks.
        tlistb (List[STrack]): The second list of tracks.

    Returns:
        List[STrack]: A combined list of tracks from both input lists, without duplicates.
    """
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: List["STrack"], tlistb: List["STrack"]) -> List["STrack"]:
    """
    Subtracts the tracks in tlistb from tlista based on track IDs.

    Args:
        tlista (List[STrack]): The list of tracks from which tracks will be removed.
        tlistb (List[STrack]): The list of tracks to be removed from tlista.

    Returns:
        List[STTrack]: The remaining tracks after removal.
    """
    stracks = {t.id: t for t in tlista}
    for t in tlistb:
        tid = t.id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(
    stracksa: List["STrack"], stracksb: List["STrack"]
) -> Tuple[List["STrack"], List["STrack"]]:
    """
    Removes duplicate tracks between two lists based on their IoU distance and track duration.

    Args:
        stracksa (List[STrack]): The first list of tracks.
        stracksb (List[STrack]): The second list of tracks.

    Returns:
        Tuple[List[STrack], List[STrack]]: The filtered track lists, with duplicates removed.
    """
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []

    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]

    return resa, resb

def remove_duplicate_stracks_with_coex(
    stracksa: List["STrack"],
    stracksb: List["STrack"],
    iou_thr: float = 0.15,
    coex_map: dict | None = None,
) -> Tuple[List["STrack"], List["STrack"]]:
    """
    Loại trùng giữa 2 danh sách dựa trên IoU + thời lượng, 
    NHƯNG KHÔNG xóa nếu hai ID đã từng đồng-tồn-tại (coexist) trong coex_map.
    """
    if not stracksa or not stracksb:
        return stracksa, stracksb

    pdist = iou_distance(stracksa, stracksb)  # 0 tốt, 1 xấu
    pairs = np.where(pdist < float(iou_thr))
    dupa, dupb = set(), set()

    for p, q in zip(*pairs):
        ta, tb = stracksa[p], stracksb[q]
        ida = int(getattr(ta, "id", -1))
        idb = int(getattr(tb, "id", -1))

        # nếu đã từng đồng-tồn-tại -> không coi là duplicate
        if coex_map is not None:
            peers = coex_map.get(ida, set())
            if idb in peers:
                continue

        # chưa/co-không coexist: chọn 1 để bỏ
        timep = getattr(ta, "frame_id", 0) - getattr(ta, "start_frame", 0)
        timeq = getattr(tb, "frame_id", 0) - getattr(tb, "start_frame", 0)

        if timep > timeq:
            dupb.add(q)
        elif timeq > timep:
            dupa.add(p)
        else:
            # tie-breaker: giữ cái có frame_id mới hơn; tiếp theo ưu tiên conf
            fpa = getattr(ta, "frame_id", 0)
            fpb = getattr(tb, "frame_id", 0)
            if fpa > fpb:
                dupb.add(q)
            elif fpb > fpa:
                dupa.add(p)
            else:
                cfa = float(getattr(ta, "conf", 0.0))
                cfb = float(getattr(tb, "conf", 0.0))
                if cfa >= cfb:
                    dupb.add(q)
                else:
                    dupa.add(p)

    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_box1 + area_box2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0


def iou_batch(bboxes1, bboxes2):
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)
    xx1 = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    yy1 = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    xx2 = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    yy2 = np.minimum(bboxes1[:, 3], bboxes2[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

def calculate_area_from_coordinates(coords):
    return (coords[2] - coords[0]) * (coords[3] - coords[1])

def calculate_intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    return interW * interH

import cv2
import numpy as np
from boxmot.trackers.botsort.botsort_track import crop_image

def compute_frame_metrics(track, det_or_bbox, img, active_tracks):
    """
    Tính toán các chỉ số chất lượng cho feature:
        - area: tỷ lệ bbox/frame
        - occlusion: mức độ che khuất so với các track khác
        - sharpness: độ nét từ crop ảnh bbox

    Args:
        track (STrack): Track hiện tại
        det_or_bbox: Có thể là:
            - STrack (có .xyxy)
            - numpy array hoặc list dạng [x1,y1,x2,y2]
        img (np.ndarray): Frame hiện tại
        active_tracks (list): Danh sách active tracks để tính occlusion

    Returns:
        (area, occlusion, sharpness)
    """

    # ✅ Lấy bbox theo định dạng numpy [x1,y1,x2,y2]
    if hasattr(det_or_bbox, "xyxy"):
        bbox = np.array(det_or_bbox.xyxy)
    else:
        bbox = np.array(det_or_bbox, dtype=float)

    # Tính area (tỷ lệ so với frame)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    frame_h, frame_w = img.shape[:2]
    area = (w * h) / (frame_w * frame_h + 1e-6)

    # ✅ Tính occlusion so với các track khác
    occlusion = 0.0
    if active_tracks:
        for t in active_tracks:
            if getattr(t, "id", None) == getattr(track, "id", None):
                continue
            other_bbox = t.xyxy
            xx1 = max(bbox[0], other_bbox[0])
            yy1 = max(bbox[1], other_bbox[1])
            xx2 = min(bbox[2], other_bbox[2])
            yy2 = min(bbox[3], other_bbox[3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            if w * h > 0:
                ratio = inter / (w * h)
                occlusion = max(occlusion, ratio)

    # ✅ Tính sharpness
    cropped = crop_image(img, bbox)
    if cropped.size > 0:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(sharpness / 1000.0, 1.0)  # normalize về [0,1]
    else:
        sharpness = 0.0

    return area, occlusion, sharpness

def embedding_distance_botsort(tracks, detections, metric="cosine"):
    M, N = len(tracks), len(detections)
    cost_matrix = np.zeros((M, N), dtype=np.float32)
    if M == 0 or N == 0:
        return cost_matrix

    det_feats_list = [getattr(d, "curr_feat", None) for d in detections]
    feat_dim = next((len(f) for f in det_feats_list if f is not None), None)
    if feat_dim is None:
        return np.ones((M, N), dtype=np.float32)

    det_feats = [
        (f if f is not None else np.zeros(feat_dim, dtype=np.float32))
        for f in det_feats_list
    ]

    tr_feats = []
    for t in tracks:
        f = getattr(t, "smooth_feat", None)
        if f is None:
            f = getattr(t, "curr_feat", None)
        if f is None or len(f) != feat_dim:
            f = np.zeros(feat_dim, dtype=np.float32)
        tr_feats.append(f)

    tr = np.asarray(tr_feats, dtype=np.float32)
    de = np.asarray(det_feats, dtype=np.float32)

    eps = 1e-6
    tr /= np.clip(np.linalg.norm(tr, axis=1, keepdims=True), eps, None)
    de /= np.clip(np.linalg.norm(de, axis=1, keepdims=True), eps, None)

    # cosine distance (như cdist cosine), ra [0, 2] nếu chưa chia 2
    cost = 1.0 - tr @ de.T
    cost = np.clip(cost, 0.0, 2.0).astype(np.float32)
    return cost

# botsort_utils.py

# botsort_utils.py
def build_feat_matrix(objs, prefer=("long_feat_mean", "smooth_feat", "curr_feat")):
    feats, valid, ref = [], [], None
    for o in objs:
        v = None
        for name in prefer:
            v = getattr(o, name, None)
            if v is not None:
                break
        feats.append(v)
        ok = v is not None
        valid.append(ok)
        if ok and ref is None:
            ref = v

    valid = np.asarray(valid, dtype=bool)
    if ref is None:
        return None, valid

    ref = np.asarray(ref, dtype=np.float32).ravel()
    D = int(ref.shape[0])
    mat = np.zeros((len(feats), D), dtype=np.float32)

    for i, v in enumerate(feats):
        if v is None:
            continue
        x = np.asarray(v, dtype=np.float32, copy=False).ravel()
        n = np.linalg.norm(x)
        # chỉ chuẩn hoá khi lệch đáng kể so với 1
        if np.isfinite(n) and n > 1e-6 and abs(n - 1.0) > 0.01:
            mat[i] = x / n
        else:
            mat[i] = x
    return mat, valid


def first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


class OptimizedOverlapCalculator:
    def __init__(self):
        self._frame = -1
        self._cache = {}

    @staticmethod
    def _pairwise_overlap_ratio(boxes):
        # boxes: (N,4) float32, trả về overlap_ratio theo diện tích hộp i: inter/area_i
        x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
        y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
        y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

        # diện tích theo hàng (area_i)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # tránh chia 0
        denom = np.clip(areas, 1e-6, None)[:, None]
        ratios = inter / denom
        np.fill_diagonal(ratios, 0.0)
        return ratios  # (N,N)

    def build_cache_for_context(self, tracks, frame_id, threshold=0.45):
        if frame_id != self._frame:
            self._cache.clear()
            self._frame = frame_id

        objs = []
        keys = []
        for t in tracks:
            if hasattr(t, "xyxy"):
                objs.append(t)
                # dùng object identity để không đụng nhau
                keys.append(id(t))

        n = len(objs)
        if n <= 1:
            for k in keys:
                self._cache[k] = {"max_ratio": 0.0, "is_occluded": False}
            return

        boxes = np.asarray([o.xyxy for o in objs], dtype=np.float32, order="C")
        ratios = self._pairwise_overlap_ratio(boxes)  # (N,N)
        max_per_row = ratios.max(axis=1) if ratios.size else np.zeros(n, np.float32)

        for k, r in zip(keys, max_per_row):
            self._cache[k] = {"max_ratio": float(r), "is_occluded": bool(r >= threshold)}

    def get_result(self, track, default_thresh=0.45):
        # Có thể gọi sau khi build_cache_for_context
        info = self._cache.get(id(track))
        if info is None:
            # fallback nếu chưa build
            return {"max_ratio": 0.0, "is_occluded": False}
        return info
