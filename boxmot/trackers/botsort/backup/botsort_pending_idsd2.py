# botsort_pending_v2.py

from pathlib import Path
from collections import deque
import numpy as np
import cv2
import torch

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.botsort_utils import (
    joint_stracks,
    sub_stracks,
    embedding_distance_botsort,
    build_feat_matrix,
    first_not_none,
)

from boxmot.utils.matching import (
    fuse_score,
    iou_distance,
    linear_assignment,
)

from boxmot.trackers.botsort.pending_manager import PendingManager
from boxmot.qdrant.qdrant_long_reid import HybridLongBank

class BotSort(BaseTracker):
    """
    BoTSORT Tracker: A tracking algorithm that combines appearance and motion-based tracking.

    Args:
        reid_weights (str): Path to the model weights for ReID.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        half (bool): Use half-precision (fp16) for faster inference.
        per_class (bool, optional): Whether to perform per-class tracking.
        track_high_thresh (float, optional): Detection confidence threshold for first association.
        track_low_thresh (float, optional): Detection confidence threshold for ignoring detections.
        new_track_thresh (float, optional): Threshold for creating a new track.
        track_buffer (int, optional): Frames to keep a track alive after last detection.
        match_thresh (float, optional): Matching cost threshold for data association.
        proximity_thresh (float, optional): IoU distance threshold for first-round association.
        appearance_thresh (float, optional): Appearance embedding distance threshold for ReID.
        cmc_method (str, optional): Method for correcting camera motion, e.g., "sof" (simple optical flow).
        frame_rate (int, optional): Video frame rate, used to scale the track buffer.
        fuse_first_associate (bool, optional): Fuse appearance and motion in the first association step.
        with_reid (bool, optional): Use ReID features for association.
    """

    def __init__(
        self,
        reid_weights        : str   = Path,
        device              : str   = torch.device,
        half                : bool  = False,
        per_class           : bool  = False,
        track_high_thresh   : float = 0.5,
        track_low_thresh    : float = 0.1,
        new_track_thresh    : float = 0.6,
        track_buffer        : int   = 30,
        match_thresh        : float = 0.8,
        proximity_thresh    : float = 0.5,
        appearance_thresh   : float = 0.25,
        cmc_method          : str   = "ecc",
        frame_rate          : int   = 30,
        fuse_first_associate: bool  = False,
        with_reid           : bool  = True,
        # Qdrant parameters
        use_qdrant          : bool  = True,
        qdrant_host         : str   = "localhost",
        qdrant_port         : int   = 6333,
        qdrant_collection   : str   ="long_term_reid",
        bank_slots          : int   = 20,
    ):
        super().__init__(per_class=per_class)
        self.lost_stracks    = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class         = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh  = track_low_thresh
        self.new_track_thresh  = new_track_thresh
        self.match_thresh      = match_thresh

        self.buffer_size   = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh  = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid         = with_reid
        if self.with_reid and reid_weights is not None:
            try:
                self.model = ReidAutoBackend(
                    weights=reid_weights, device=device, half=half
                ).model
            except Exception as e:
                print(f"Warning: Failed to load ReID model: {e}")
                self.with_reid = False
                self.model = None
        else:
            self.model = None
            if with_reid and reid_weights is None:
                print("Warning: with_reid=True but reid_weights=None , disabling ReID")
                self.with_reid = False
        
        self.cmc = get_cmc_method(cmc_method)()
        self.fuse_first_associate = fuse_first_associate    
        
        self.use_dynamic_weights = True
        self.w_motion_base       = 0.6
        self.w_motion_start_dw   = 0.55 
        
        self.dw_start_frames = 0    
        self.dw_step_frames  = 20
        self.dw_step_delta   = 0.05
        self.dw_app_split    = 0.5
        
        self.long_update_cos_thresh = 0.12   # Cho phép cập nhật đặc trưng
        self.max_obs = 50
        
        reid_dim = getattr(self.model, "feature_dim", 512) if self.with_reid and self.model else None
        self.long_bank_topk = 5   # số điểm trong bank để pooling
        
        self.pending_manager = PendingManager(
            kalman_filter              =self.kalman_filter,
            min_lost_matches_to_promote=5,
            promotion_deadline         =15,
            iou_thresh                 =0.5,
            appearance_thresh          =0.12,
            match_thresh               =0.7,
            use_dynamic_weights        =self.use_dynamic_weights,
            w_motion_base              =self.w_motion_base,
            w_motion_start_dw          =self.w_motion_start_dw,
            dw_start_frames            =self.dw_start_frames,
            dw_step_frames             =self.dw_step_frames,
            dw_step_delta              =self.dw_step_delta,
            reid_dim                   =reid_dim,
            dw_app_split               =self.dw_app_split,
            promote_min_frames_for_lost=5,
            proto_provider             =self._bank_centroid_for_track,
            vectors_provider           =self._bank_vectors_for_track,
            long_bank_topk             =self.long_bank_topk,
            debug_pending_lost         =True
        )
        
        self.occlusion_overlap_thresh = 0.45 # Diện tích bị che -> không cập nhật
        
        self.coex_map = {}   # id -> set các id đã từng cùng xuất hiện (active_track và lost_track)                
                
        # --- Long Feature Bank (REPLACE) ---
        import uuid as _uuid
        self.run_uid    = _uuid.uuid4().hex[:8]
        self.bank_slots = int(bank_slots)

        # cache & ring-slot cho từng track id
        self._slot_pos           = {}
        self._bank_proto_cache   = {}
        self._bank_proto_ttl     = 5

        self.long_bank = None
        if self.with_reid and reid_dim:
            try:
                self.long_bank = HybridLongBank(
                    use_qdrant = bool(use_qdrant),
                    host       = qdrant_host,
                    port       = int(qdrant_port),
                    collection = qdrant_collection,
                    dim        = int(reid_dim),
                    slots      = self.bank_slots,
                )
                print(f"[LongBank] backend={self.long_bank.backend_name}, slots={self.bank_slots}")
            except Exception as e:
                print(f"[LongBank] init failed, disabled. Reason: {e}")
                self.long_bank = None
                
        self._frame_cache_bank = {} #  {tid: (frame_id, V)} cache V của từng track 1-2 frame
        # -----------------------------------
        
        self._idswap_locked_pairs = set()  
        self._idswap_locked_ids   = set()
        
        # --- suspect/holder voting config --- 
        self.suspect_consec     = 7  # tự cost > ngưỡng trong 7 frame liên tiếp
        self.holder_vote_win    = 10 # cửa sổ 10 frame để vote holder
        self.holder_vote_need   = 5  # cần ≥5 lần pass trong cửa sổ để xác nhận
        self.no_motion_cooldown = 5  # số frame chặn association với track vừa bị tách
        self.use_cooldown       = False   # <--- Tắt cooldown tạm thời
        
            
    # ------------- Long Bank Helpers ---------------
    def _bank_next_slot(self, tid: int) -> int:
        pos = self._slot_pos.get(int(tid), 0)
        self._slot_pos[int(tid)] = (pos + 1) % self.bank_slots
        return pos
    
    def _bank_add_feature(self, track: STrack, feat: np.ndarray):
        """Push one vector into bank ring-buffer (if available)."""
        if self.long_bank is None or (not self.with_reid) or feat is None:
            return
        try:
            v = np.asarray(feat, dtype=np.float32).ravel()
            if not np.isfinite(v).all() or v.size == 0:
                return
            slot = self._bank_next_slot(track.id)
            extra = {"frame_id": int(self.frame_count), "cls": int(getattr(track, "cls", 0))}
            self.long_bank.add_ring(self.run_uid, int(track.id), v, slot, extra=extra)
            # invalidate centroid cache for this track (recompute soon)
            self._bank_proto_cache.pop(int(track.id), None)
        except Exception as e:
            print(f"[LongBank] add feature failed for id={getattr(track,'id',-1)}: {e}")

    def _bank_centroid_for_track(self, track: STrack):
        """
        Get normalized centroid from bank with small TTL cache.
        Fallback: track.long_feat_mean / track.smooth_feat / None
        """
        tid = int(getattr(track, "id", -1))
        if tid < 0 or self.long_bank is None:
            # fallback to local fields
            for key in ("long_feat_mean", "smooth_feat", "curr_feat"):
                v = getattr(track, key, None)
                if v is not None:
                    v = np.asarray(v, dtype=np.float32).ravel()
                    n = np.linalg.norm(v) + 1e-6
                    return (v / n), True
            return None, False

        # cache
        cache = self._bank_proto_cache.get(tid)
        if cache is not None:
            last_f, vec = cache
            if (self.frame_count - last_f) <= self._bank_proto_ttl and vec is not None:
                return vec, True

        # fetch from bank
        try:
            V, _ = self.long_bank.get_vectors_with_payload(self.run_uid, tid)
            if V is not None and len(V) > 0:
                c = V.mean(axis=0)
                n = np.linalg.norm(c) + 1e-6
                c = (c / n).astype(np.float32)
                self._bank_proto_cache[tid] = (self.frame_count, c)
                return c, True
        except Exception as e:
            print(f"[LongBank] fetch centroid failed for id={tid}: {e}")

        # fallback to local
        for key in ("long_feat_mean", "smooth_feat", "curr_feat"):
            v = getattr(track, key, None)
            if v is not None:
                v = np.asarray(v, dtype=np.float32).ravel()
                n = np.linalg.norm(v) + 1e-6
                c = (v / n).astype(np.float32)
                self._bank_proto_cache[tid] = (self.frame_count, c)
                return c, True

        return None, False
    
    def _build_tracks_bank_matrix(self, tracks):
        """
        Build matrix T (M x D) from bank centroids for given tracks.
        Returns T, tvalid(bool mask)
        """
        M = len(tracks)
        if M == 0:
            return None, np.zeros((0,), dtype=bool)

        vecs = []
        valid = []
        dim  = None
        for t in tracks:
            v, ok = self._bank_centroid_for_track(t)
            if ok and v is not None:
                v = np.asarray(v, dtype=np.float32).ravel()
                if dim is None:
                    dim = v.size
                if v.size != dim:
                    # dimension mismatch -> invalidate
                    v = None
                    ok = False
            vecs.append(v)
            valid.append(bool(ok))

        if dim is None:
            return None, np.asarray(valid, dtype=bool)

        T = np.zeros((M, dim), dtype=np.float32)
        for i, v in enumerate(vecs):
            if v is not None:
                T[i, :] = v
        return T, np.asarray(valid, dtype=bool)    

    def allow_update(self, long_val, track, other_tracks=None):
        """Check if track updates are allowed based on vảious conditions."""
        long_th = float(getattr(self, "long_update_cos_thresh", 0.15))
        occ_th  = float(getattr(self, "occlusion_overlap_thresh", 0.45))
        
        long_ok = True
        if long_val is not None:
            long_ok = long_val <= long_th
            
        occluded = False
        if other_tracks:
            try:
                occluded = self._occluded_heavily(track, [t for t in other_tracks if t.id != track.id], thresh=occ_th)
            except Exception:
                occluded = False
                
        allow_short = True
        allow_long  = bool(allow_short and long_ok and(not occluded))
        
        return allow_short, allow_long, occluded

    def _occlusion_context(self, exclude_id=None, cls_ref=None, predict_pending=True):
        """Lấy danh sách đối tượng có thể gây che khuất: active (Tracked) + pending."""
        ctx = []
        # Active đang Tracked
        try:
            for t in self.active_tracks:
                if t.state == TrackState.Tracked and getattr(t, "id", None) != exclude_id:
                    ctx.append(t)
        except Exception:
            pass
        # Pending
        try:
            p_list = getattr(self.pending_manager, "pending_tracks", [])
            if predict_pending:
                for p in p_list:
                    try:
                        p.predict()
                    except Exception:
                        pass
            for p in p_list:
                if getattr(p, "id", None) != exclude_id:
                    ctx.append(p)  # p đã có .xyxy và .id
        except Exception:
            pass

        # per-class filtering
        if cls_ref is not None:
            ctx = [o for o in ctx if getattr(o, "cls", None) == cls_ref]

        return ctx
    
    def _update_coexistence_simple(self, tracks):
        """Record all pairs of IDs that coexist in current frame."""
        try:
            live = [t for t in tracks if t.state == TrackState.Tracked]
            lock = getattr(self, "_idswap_locked_ids", set())  # IDs thuộc cụm nghi ngờ ở frame này
            n = len(live)
            for i in range(n):
                ida = int(live[i].id)
                self.coex_map.setdefault(ida, set())
                for j in range(i + 1, n):
                    idb = int(live[j].id)
                    if idb == ida:
                        continue
                    # chỉ chặn coexistence giữa các ID đều đang bị nghi ngờ
                    if ida in lock or idb in lock:
                        continue
                    self.coex_map.setdefault(idb, set())
                    self.coex_map[ida].add(idb)
                    self.coex_map[idb].add(ida)
        except Exception as e:
            print(f"Error in coexistence update: {e}")
                
    def _purge_from_coex(self, ids):
        """Remove all traces of given ids from coexistence map."""
        if ids is None:
            return
        try:
            for tid in list(ids):
                tid = int(tid)
                peers = self.coex_map.pop(tid, None)
                if peers:
                    for p in list(peers):
                        s = self.coex_map.get(int(p))
                        if s is not None:
                            s.discard(tid)
        except Exception as e:
            print(f"Error purging coexistence: {e}")
    
    def _bank_vectors_for_track(self, track):
        """Lấy toàn bộ vectors V (N_i, D) từ bank cho track.id, có cache ngắn theo frame."""
        if self.long_bank is None or (not self.with_reid) or getattr(track, "id", None) is None:
            return None
        tid = int(track.id)
        cached = self._frame_cache_bank.get(tid)
        if cached is not None:
            last_f, Vc = cached
            if (self.frame_count - last_f) <= 1:
                return Vc
        try:
            V, _ = self.long_bank.get_vectors_with_payload(self.run_uid, tid)
            if V is not None and len(V) > 0:
                # V đã là L2-norm từ lúc add_ring
                self._frame_cache_bank[tid] = (self.frame_count, V)
                return V
        except Exception:
            pass
        return None

    def _track_det_topk_cost_row(self, V, D, dvalid, k=8):
        """
        Tính cost cho 1 track (với bank V) tới tất cả detections D (N_det, D).
        Trả về: row cost shape (N_det,), cost = 0.5*(1 - mean_topk_cos).
        """
        N = 0 if D is None else D.shape[0]
        if V is None or N == 0:
            return np.ones((N,), dtype=np.float32)
        try:
            # S = V @ D^T => (N_i, N_det)
            S = V @ D.T
            np.clip(S, -1.0, 1.0, out=S)
            k_use = min(int(k), int(S.shape[0]))
            if k_use <= 0:
                return np.ones((N,), dtype=np.float32)
            # Top-k theo cột (theo detection): n_i nhỏ (≤ bank_slots), sort là đủ nhanh
            topk = np.sort(S, axis=0)[-k_use:, :]   # (k_use, N_det)
            sims = topk.mean(axis=0)                # (N_det,)
            cost = 0.5 * (1.0 - sims)
            # Vô hiệu các detection không có feat hợp lệ
            cost = np.where(dvalid, cost, 1.0).astype(np.float32, copy=False)
            return cost
        except Exception:
            return np.ones((N,), dtype=np.float32)

    def _compute_long_cost(self, tracks, dets):
        """Long-term cost với top-k pooling trên vector bank: cost = 0.5*(1 - mean_topk_cos)."""
        M, N = len(tracks), len(dets)
        if M == 0 or N == 0:
            return np.ones((M, N), dtype=np.float32)
        try:
            # Lấy ma trận đặc trưng detections (ưu tiên curr_feat -> smooth_feat)
            
            D, dvalid = build_feat_matrix(dets  , ("curr_feat", "smooth_feat"))
            if D is None:
                return np.ones((M, N), dtype=np.float32)
            
            D = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-6, None)
            
            C = np.ones((M, N), dtype=np.float32)
            for i, t in enumerate(tracks):
                V = self._bank_vectors_for_track(t)  # (N_i, D) hoặc None
                row = self._track_det_topk_cost_row(V, D, dvalid, k=self.long_bank_topk)
                C[i, :] = row
            return np.clip(C, 0.0, 1.0, out=C)
        
        except Exception as e:
            print(f"Error computing long cost (top-k): {e}")
            return np.ones((M, N), dtype=np.float32)
            
    def _occluded_heavily(self, track, other_tracks, thresh=None):
        """Check if track is heavily occluded by other tracks."""
        if thresh is None:
            thresh = getattr(self, "occlusion_overlap_thresh", 0.5)
        try:
            ratio = self._max_overlap_ratio(track, other_tracks)
            return ratio >= float(thresh)
        except Exception:
            return False            

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes."""
        try:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter <= 0:
                return 0.0
                
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter
            
            return inter / max(union, 1e-6)
        except Exception:
            return 0.0

    def _cos_cost_simple(self, a, b):
        if a is None or b is None: 
            return 1.0
        
        a = np.asarray(a, np.float32).ravel()
        b = np.asarray(b, np.float32).ravel()
        na = np.linalg.norm(a) + 1e-6
        nb = np.linalg.norm(b) + 1e-6
        return 0.5 * (1.0 - float(np.dot(a, b)) / (na * nb))

    def _feat_of(self, t):
        return first_not_none(getattr(t, "curr_feat", None), getattr(t, "smooth_feat", None))
    
    def _try_match_single_to_lost_long_only(self, single_track: STrack, lost_pool, img=None) -> bool:
        """
        Reclaim ID cho 'single_track' từ lost_pool chỉ bằng long-term
        """
        try:
            if not lost_pool:
                return False

            # 1) Tính long cost giữa LOST (tracks) và SINGLE (như 1 detection)
            #    _compute_long_cost(tracks, dets) => shape (len(lost_pool), 1)
            C = self._compute_long_cost(lost_pool, [single_track])
            C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0).ravel()  # (N,)

            # 2) Chặn các LOST từng "đồng xuất hiện" với ID hiện tại (coexistence guard)
            curr_id = int(getattr(single_track, "id", -1))
            coex_block = self.coex_map.get(curr_id, set())
            for j, lt in enumerate(lost_pool):
                lid = int(getattr(lt, "id", -1))
                if lid in coex_block:
                    C[j] = 1.0  # block

            # 3) Chọn best theo ngưỡng long-term
            long_thr = float(getattr(self, "appearance_thresh", 0.25))
            jbest = int(np.argmin(C))
            if C[jbest] <= long_thr:
                lost_trk = lost_pool[jbest]

                # re-activate LOST bằng đo hiện tại của single_track
                lost_trk.re_activate(single_track, self.frame_count, new_id=False, img=img)

                # Nếu single đang nằm trong active thì gỡ ra
                if single_track in getattr(self, "active_tracks", []):
                    self.active_tracks.remove(single_track)

                # Đẩy 1 vector an toàn về bank (tuỳ chọn)
                if self.with_reid:
                    feat_for_bank = getattr(lost_trk, "curr_feat", None) or getattr(single_track, "curr_feat", None)
                    if feat_for_bank is not None:
                        proto_vec, ok = self._bank_centroid_for_track(lost_trk)
                        safe = True
                        if ok and proto_vec is not None:
                            safe = (self._cos_cost_simple(feat_for_bank, proto_vec)
                                    <= float(getattr(self, "long_update_cos_thresh", 0.15)))
                        if safe and (int(lost_trk.id) not in getattr(self, "_idswap_locked_ids", set())):
                            self._bank_add_feature(lost_trk, feat_for_bank)
                return True

            return False
        except Exception:
            return False
    
    def _clusters_suspect_holder(self, live_tracks):
        """
        Trả về:
        - clusters: list[list[STrack]] được xác nhận bằng voting suspect-holder
        - singles_to_split: list[STrack] hết cửa sổ mà không đủ vote -> cần tách ngay
        """
        n = len(live_tracks)
        if n == 0:
            return [], []

        # Matrix prototype (ID) và feature hiện tại
        P, pvalid = self._build_tracks_bank_matrix(live_tracks)
        F, fvalid = build_feat_matrix(live_tracks, ("curr_feat", "smooth_feat"))
        if P is None or F is None:
            return [], []

        C = 0.5 * (1.0 - (F @ P.T))
        np.clip(C, 0.0, 1.0, out=C)
        mask = pvalid[:, None] & fvalid[None, :]
        C = np.where(mask, C, 1.0).astype(np.float32, copy=False)

        # cập nhật trạng thái suspect/holder theo cửa sổ thời gian
        pairs, singles_idx = self._update_suspect_holder_state(live_tracks, C, pvalid, fvalid)

        # build cluster từ các cặp đã được xác nhận (đồ thị vô hướng)
        adj = {i: set() for i in range(n)}
        for i, j in pairs:
            adj[i].add(j)
            adj[j].add(i)

        visited = set()
        clusters = []
        for i in range(n):
            if i in visited or not adj[i]:
                continue
            comp = []
            stack = [i]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        stack.append(v)
            clusters.append([live_tracks[k] for k in comp])

        singles_to_split = [live_tracks[k] for k in singles_idx]
        return clusters, singles_to_split
    
    def _update_suspect_holder_state(self, tracks, C, pvalid, fvalid):
        """
        Quản lý đếm suspect và voting holder trong cửa sổ.
        - Soft-pause khi occluded, nhưng có timeout để không deadlock.
        Trả về:
        pairs: list[(i_idx, j_idx)]
        singles_idx: list[int]
        """
        pairs = []
        singles_idx = []

        th_self = float(self.long_update_cos_thresh)
        th_hold = float(self.appearance_thresh)
        consec  = int(getattr(self, "suspect_consec", 5))
        win     = int(getattr(self, "holder_vote_win", 10))
        need    = int(getattr(self, "holder_vote_need", 5))
        max_occ = int(getattr(self, "max_occlusion_frames", 20))  # soft-timeout khi che

        for i, t in enumerate(tracks):
            # init state
            if not hasattr(t, "_suspect_run"):       t._suspect_run = 0
            if not hasattr(t, "_in_holder_window"):  t._in_holder_window = False
            if not hasattr(t, "_holder_votes"):      t._holder_votes = {}
            if not hasattr(t, "_holder_win_left"):   t._holder_win_left = 0
            if not hasattr(t, "_grace_period"):      t._grace_period = 0
            if not hasattr(t, "_occlusion_frames"):  t._occlusion_frames = 0

            # --- PAUSE khi occluded (có timeout) ---
            is_occ = bool(getattr(t, "_is_occluded", False))
            t._is_occluded = False  # reset cờ theo frame
            if is_occ:
                t._occlusion_frames = min(max_occ, t._occlusion_frames + 1)
            else:
                t._occlusion_frames = 0

            # soft pause trong thời gian che < max_occ
            if is_occ and t._occlusion_frames < max_occ:
                continue
            # hết timeout -> xử lý bình thường

            # --- tín hiệu appearance ---
            if not (pvalid[i] and fvalid[i]):
                t._suspect_run = 0
                t._in_holder_window = False
                t._holder_votes.clear()
                t._holder_win_left = 0
                t._grace_period = 0
                continue

            self_cost = float(C[i, i])

            if self_cost > th_self:
                t._suspect_run += 1
            else:
                # tự hồi
                t._suspect_run = 0
                t._in_holder_window = False
                t._holder_votes.clear()
                t._holder_win_left = 0
                t._grace_period = 0
                continue

            if t._suspect_run >= consec:
                # mở/duy trì cửa sổ vote
                if not t._in_holder_window:
                    t._in_holder_window = True
                    t._holder_votes = {}
                    t._holder_win_left = win
                else:
                    t._holder_win_left = max(0, t._holder_win_left - 1)

                # vote các holder ứng viên
                for j, other in enumerate(tracks):
                    if j == i:
                        continue
                    hid = int(getattr(other, "id", -1))
                    dq = t._holder_votes.get(hid)
                    if dq is None:
                        dq = deque(maxlen=win)
                    dq.append(1 if C[i, j] <= th_hold else 0)
                    t._holder_votes[hid] = dq

                # tìm holder thắng
                best_hid, best_score = None, 0
                for hid, dq in t._holder_votes.items():
                    s = sum(dq)
                    if s > best_score:
                        best_score, best_hid = s, hid

                if best_hid is not None and best_score >= need:
                    # map holder id -> index
                    j_idx = next((k for k, o in enumerate(tracks) if int(getattr(o, "id", -1)) == best_hid), None)
                    if j_idx is not None:
                        pairs.append((i, j_idx))
                    # reset sau khi xác nhận
                    t._in_holder_window = False
                    t._holder_votes.clear()
                    t._holder_win_left = 0
                    t._suspect_run = 0
                    t._grace_period = 0
                elif t._holder_win_left == 0:
                    # hết cửa sổ mà chưa đủ phiếu
                    if self_cost <= th_self:
                        # tự hồi
                        t._suspect_run = 0
                        t._in_holder_window = False
                        t._holder_votes.clear()
                        t._holder_win_left = 0
                        t._grace_period = 0
                    else:
                        if t._grace_period <= 0:
                            t._grace_period = 3  # đặt lần đầu
                        else:
                            t._grace_period -= 1
                            if t._grace_period == 0:
                                singles_idx.append(i)

        # TRẢ VỀ Ở CUỐI HÀM
        return pairs, singles_idx
    
    def _apply_idswap_cluster(
        self,
        cluster_tracks,
        new_tracks_buf,
        lost_stracks,
        activated_stracks,
        refind_stracks,
        img=None
    ):
        """
        Hai pha để tránh trùng ID tạm thời trong cụm:
        - Pha 1: xử lý toàn bộ loser (mark_lost/remove/reclaim/new_track)
        - Pha 2: gán ID cho winner sau khi loser đã rời active
        """
        m = len(cluster_tracks)
        if m == 0:
            return

        P, pvalid = self._build_tracks_bank_matrix(cluster_tracks)
        F, fvalid = build_feat_matrix(cluster_tracks, ("curr_feat", "smooth_feat"))
        if P is None or F is None:
            return

        C = 0.5 * (1.0 - (F @ P.T))
        np.clip(C, 0.0, 1.0, out=C)
        mask = pvalid[:, None] & fvalid[None, :]
        C = np.where(mask, C, 1.0).astype(np.float32, copy=False)

        matches, u_r, u_c = linear_assignment(C, thresh=self.appearance_thresh)

        new_ids = [None] * m
        assigned_cost = [1.0] * m
        for r, c in matches:
            new_ids[r] = int(cluster_tracks[c].id)
            assigned_cost[r] = float(C[r, c])

        ath = float(self.long_update_cos_thresh * 1.2)

        # --- Pha 1: xử lý LOSER trước
        lost_pool_now = joint_stracks(self.lost_stracks, lost_stracks)

        for r in range(m):
            # loser: không được gán hoặc cost > ath
            if new_ids[r] is None or assigned_cost[r] > ath:
                t = cluster_tracks[r]

                # Ưu tiên reclaim long-only về LOST
                if self._try_match_single_to_lost_long_only(t, lost_pool_now, img=img):
                    if t.state != TrackState.Lost:
                        t.mark_lost()
                        t._just_lost_in_frame = self.frame_count
                        lost_stracks.append(t)

                    # loại khỏi active/buffers
                    if t in self.active_tracks:
                        self.active_tracks.remove(t)
                    if activated_stracks is not None:
                        activated_stracks[:] = [a for a in activated_stracks if a is not t]
                    if refind_stracks is not None:
                        refind_stracks[:] = [a for a in refind_stracks if a is not t]

                    if getattr(self, "use_cooldown", False):
                        t.no_motion_until = self.frame_count + getattr(self, "no_motion_cooldown", 5)
                    continue

                # Không reclaim được -> mark_lost + xếp lịch tạo track mới
                if t.state != TrackState.Lost:
                    t.mark_lost()
                    t._just_lost_in_frame = self.frame_count
                    lost_stracks.append(t)

                if t in self.active_tracks:
                    self.active_tracks.remove(t)

                if activated_stracks is not None:
                    activated_stracks[:] = [a for a in activated_stracks if a is not t and getattr(a, "id", None) != getattr(t, "id", None)]
                if refind_stracks is not None:
                    refind_stracks[:] = [a for a in refind_stracks if a is not t and getattr(a, "id", None) != getattr(t, "id", None)]

                if getattr(self, "use_cooldown", False):
                    t.no_motion_until = self.frame_count + getattr(self, "no_motion_cooldown", 5)

                # chuẩn bị det_like/feat để caller activate sau
                f = self._feat_of(t)
                if f is not None:
                    f = f.astype(np.float32, copy=False)
                    f /= (np.linalg.norm(f) + 1e-6)
                det_like = np.array([*t.xyxy, getattr(t, 'conf', 1.0), getattr(t, 'cls', 0), -1], dtype=np.float32)
                new_tracks_buf.append((det_like, f))

        # --- Pha 2: gán ID cho WINNER sau khi LOSER đã rời active
        for r in range(m):
            if new_ids[r] is not None and assigned_cost[r] <= ath:
                cluster_tracks[r].id = new_ids[r]

    def _handle_single_suspect_recovery(self, track):
        """
        Cố gắng khôi phục ID gốc cho track suspect đơn lẻ
        Trả về True nếu có thể recovery, False nếu cần tạo mới
        """
        # Lấy prototype gốc từ bank
        proto, proto_valid = self._bank_centroid_for_track(track)
        if not proto_valid or proto is None:
            return False
            
        # Lấy feature hiện tại
        curr_feat = self._feat_of(track)
        if curr_feat is None:
            return False
        
        # Tính cost với chính prototype gốc của nó
        self_cost = self._cos_cost_simple(curr_feat, proto)
        
        # Nếu cost không quá tệ, cho phép recovery
        recovery_thresh = self.long_update_cos_thresh * 1.2  # Nới lỏng hơn một chút
        
        if self_cost <= recovery_thresh:
            # Reset trạng thái suspect và giữ nguyên ID
            track._suspect_run = 0
            track._in_holder_window = False
            track._holder_votes.clear()
            track._holder_win_left = 0
            return True
        
        return False

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        """Main tracking update function."""
        self.check_inputs(dets, img, embs)
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # --- 1. Preprocess detections ---
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        if self.with_reid and self.model is not None and embs is None:
            try:
                features_high = self.model.get_features(dets_first[:, 0:4], img)
            except Exception as e:
                print(f"Error extracting features: {e}")
                features_high = []
        else:
            features_high = embs_first if embs_first is not None else []

        # Extract second tier features
        features_second = None
        if self.with_reid and self.model is not None and len(dets_second) > 0 and embs is None:
            try:
                features_second = self.model.get_features(dets_second[:, 0:4], img)
            except Exception as e:
                print(f"Error extracting second tier features: {e}")
                features_second = None
        
        # Create detections
        detections = self._create_detections(dets_first, features_high)

        # Separate unconfirmed and active tracks
        unconfirmed, active_tracks = self._separate_tracks()

        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # --- 2. First association ---
        matches_first, u_track_first, u_detection_first = self._first_association(
            dets, dets_first, active_tracks, unconfirmed, img,
            detections, activated_stracks, refind_stracks, strack_pool,
        )

        # --- 3. Second association ---
        matches_second, u_track_second, u_detection_second = self._second_association(
            dets_second, activated_stracks, lost_stracks,
            refind_stracks, u_track_first, strack_pool, img,
            features_second=features_second,
        )

        # === Collect unmatched detections
        feats_for_pending = []
        raw_unmatched_dets = []
        
        # From first tier (high confidence)
        for idx in u_detection_first:
            det_box = dets_first[idx]
            raw_unmatched_dets.append(det_box)
            if self.with_reid and idx < len(features_high):
                f = features_high[idx].astype(np.float32, copy=False)
                f /= (np.linalg.norm(f) + 1e-6)
                feats_for_pending.append(f)
            else:
                feats_for_pending.append(None)
                
        # From second tier (low confidence)
        for idx in u_detection_second:
            raw_unmatched_dets.append(dets_second[idx])
            if self.with_reid and (features_second is not None) and idx < len(features_second):
                f = features_second[idx].astype(np.float32, copy=False)
                f /= (np.linalg.norm(f) + 1e-6)
                feats_for_pending.append(f)
            else:
                if self.with_reid and self.model is not None and embs is None:
                    try:
                        roi = dets_second[idx, 0:4]
                        f = self.model.get_features(np.asarray([roi]), img)[0]
                        f = f.astype(np.float32, copy=False)
                        f /= (np.linalg.norm(f) + 1e-6)
                        feats_for_pending.append(f)
                    except Exception:
                        feats_for_pending.append(None)
                else:
                    feats_for_pending.append(None)

        # --- 4. Suspect–Holder voting & cluster ---
        live_tracks = [t for t in self.active_tracks if t.state == TrackState.Tracked]

        clusters, singles_to_split = self._clusters_suspect_holder(live_tracks)

        # khoá coexistence cho các ID trong cluster xác nhận
        if clusters:
            self._idswap_locked_ids = set()
            for cl in clusters:
                for t in cl:
                    if getattr(t, "id", None) is not None:
                        self._idswap_locked_ids.add(int(t.id))

            # chạy assignment theo cụm và TẠO TRACK MỚI NGAY cho phần vượt ngưỡng
            new_tracks_buf = []  # (det_like, feat_norm)
            for cl in clusters:
                self._apply_idswap_cluster(cl, new_tracks_buf, lost_stracks, activated_stracks=activated_stracks, refind_stracks=refind_stracks, img=img)

            # vệ sinh buffers theo id các track vừa bị đấy sang lost
            if lost_stracks:
                _lost_ids = {int(getattr(t, "id", -1)) for t in lost_stracks if getattr(t, "id", None) is not None}
                if _lost_ids:
                    activated_stracks[:] = [a for a in activated_stracks if int(getattr(a, "id", -1)) not in _lost_ids]
                    refind_stracks[:]    = [a for a in refind_stracks    if int(getattr(a, "id", -1)) not in _lost_ids]
                    # và đảm bảo chúng cũng không còn trong active_tracks
                    self.active_tracks[:] = [a for a in self.active_tracks if int(getattr(a, "id", -1)) not in _lost_ids]
                    
            for det_like, feat in new_tracks_buf:
                trk = STrack(det_like, feat, max_obs=self.max_obs)
                trk.activate(self.kalman_filter, self.frame_count, img=img)
                activated_stracks.append(trk)
                if self.with_reid and feat is not None:
                    self._bank_add_feature(trk, feat)

        lost_pool = joint_stracks(self.lost_stracks, lost_stracks)
        
        # xử lý singleton hết cửa sổ mà không đủ vote -> mark_lost + tạo track mới
        for t in singles_to_split:
            # 1) thử self-recovery (giữ nguyên)
            if self._handle_single_suspect_recovery(t):
                continue
            
            # 2) long-only reclaim với LOST
            if self._try_match_single_to_lost_long_only(t, lost_pool, img=img):
                # đã re-activate xong -> đẩy track sai vào Lost (giống luồng tạo ID mới)
                if t.state != TrackState.Lost:
                    t.mark_lost()
                    t._just_lost_in_frame = self.frame_count
                    lost_stracks.append(t)
                
                # tránh lẫn trong active/buffers
                if t in self.active_tracks:
                    self.active_tracks.remove(t)
                    
                # cooldown nếu bật
                if getattr(self, "use_cooldown", False):
                    t.no_motion_until = self.frame_count + getattr(self, "no_motion_cooldown", 5)
                continue
            
            # 3) không khớp -> mark_lost + tạo track mới như cũ
            if t.state != TrackState.Lost:
                t.mark_lost()
                lost_stracks.append(t)
            
            if t in self.active_tracks:
                self.active_tracks.remove(t)
            
            if getattr(self, "use_cooldown", False):
                    t.no_motion_until = self.frame_count + getattr(self, "no_motion_cooldown", 5)

            f = self._feat_of(t)
            if f is not None:
                f = f.astype(np.float32, copy=False)
                f /= (np.linalg.norm(f) + 1e-6)
                
            det_like = np.array([*t.xyxy, getattr(t, 'conf', 1.0), getattr(t, 'cls', 0), -1], dtype=np.float32)
            new_trk = STrack(det_like, f, max_obs=self.max_obs)
            new_trk.activate(self.kalman_filter, self.frame_count, img=img)
            activated_stracks.append(new_trk)
            if self.with_reid and f is not None:
                self._bank_add_feature(new_trk, f)
        
        # --- 5. Update Pending ---
        live_tracks = [t for t in self.active_tracks if t.state == TrackState.Tracked]
        other_tracks = live_tracks + list(activated_stracks) + [
            STrack(det, f) for det, f in zip(raw_unmatched_dets, feats_for_pending) if det is not None
        ]        
        
        unmatched_dets, unmatched_feats, merged_to_lost = self.pending_manager.update_pending(
            raw_unmatched_dets,
            feats       =feats_for_pending, 
            lost_tracks =lost_pool,
            frame_id    =self.frame_count, 
            with_reid   =self.with_reid,
            img         =img,
            other_tracks=other_tracks
        )
            
        # --- 6. Add new pending ---
        if unmatched_dets:
            self.pending_manager.add_pending(unmatched_dets, self.frame_count, feats=unmatched_feats, img=img)
            
        # --- 7. Promote pending ---
        promotables = self.pending_manager.promote_pending(self.frame_count)

        if promotables:
            lost_map = {t.id: t for t in self.lost_stracks}
            min_frames = getattr(self.pending_manager, "promote_min_frames_for_lost", 5)
            
            for p_track in promotables:
                try:
                    # Get candidate IDs from pending track
                    cand_ids = [lid for lid, st in getattr(p_track, "cand_lost", {}).items() 
                                if st is not None and st.get("frames", 0) >= min_frames and lid in lost_map]

                    if len(cand_ids) == 0:
                        # No lost candidates - create new track
                        feat_for_strack = (
                            p_track.smooth_feat if getattr(p_track, "smooth_feat", None) is not None
                            else getattr(p_track, "curr_feat", None)
                        )
                        if feat_for_strack is not None:
                            feat_for_strack = feat_for_strack / np.clip(np.linalg.norm(feat_for_strack), 1e-6, None)

                        new_track = STrack(np.array([*p_track.xyxy, p_track.conf, p_track.cls, -1]), feat=feat_for_strack)
                        new_track.activate(self.kalman_filter, self.frame_count, img=img)
                        activated_stracks.append(new_track)
                        
                        self._bank_add_feature(new_track, feat_for_strack)
                        continue

                    # ID Fragment Resolution
                    canonical_id, merge_ids = self.pending_manager.resolve_id_fragments(
                        cand_ids, lost_map, p_track, self.frame_count
                    )
                
                    if canonical_id is None:
                        # Cannot resolve - create new track
                        feat_for_strack = (
                            p_track.smooth_feat if getattr(p_track, "smooth_feat", None) is not None
                            else getattr(p_track, "curr_feat", None)
                        )
                        if feat_for_strack is not None:
                            feat_for_strack = feat_for_strack / np.clip(np.linalg.norm(feat_for_strack), 1e-6, None)

                        new_track = STrack(np.array([*p_track.xyxy, p_track.conf, p_track.cls, -1]), feat=feat_for_strack)
                        new_track.activate(self.kalman_filter, self.frame_count, img=img)
                        activated_stracks.append(new_track)
                        self._bank_add_feature(new_track, feat_for_strack)
                        continue

                    # 3. Coexistence filtering for canonical_id
                    coex = self.coex_map
                    coex_set = coex.get(int(canonical_id), set())
                
                    # Check if canonical conflicts with any merge_id
                    conflicted = any((other != canonical_id) and (other in coex_set) 
                                     for other in ([canonical_id] + merge_ids))
                
                    if conflicted:
                    # Conflict after resolve - create new track
                        feat_for_strack = (
                            p_track.smooth_feat if getattr(p_track, "smooth_feat", None) is not None
                            else getattr(p_track, "curr_feat", None)
                        )
                        if feat_for_strack is not None:
                            feat_for_strack = feat_for_strack / np.clip(np.linalg.norm(feat_for_strack), 1e-6, None)

                        new_track = STrack(np.array([*p_track.xyxy, p_track.conf, p_track.cls, -1]), feat=feat_for_strack)
                        new_track.activate(self.kalman_filter, self.frame_count, img=img)
                        activated_stracks.append(new_track)
                        continue
                    
                    # Re-activate với canonical ID
                    root_trk = lost_map[canonical_id]
                    root_trk.re_activate(p_track, self.frame_count, new_id=False, img=img)
                    refind_stracks.append(root_trk)
                    del lost_map[canonical_id]

                    # --- Push 1 vector về bank sau khi re-active canonical ID ---
                    try:
                        if self.long_bank is not None and self.with_reid:
                            # Ưu tiên feat sau re-activate (đã đi qua pipeline chuẩn hoá của STrack)
                            feat_for_bank = getattr(root_trk, "curr_feat", None)

                            # Fallback: lấy từ pending nếu cần
                            if feat_for_bank is None:
                                feat_for_bank = (getattr(p_track, "smooth_feat", None) 
                                                or getattr(p_track, "curr_feat", None))

                            if feat_for_bank is not None:
                                proto_vec, ok = self._bank_centroid_for_track(root_trk)
                                safe_to_push = True
                                if ok and proto_vec is not None:
                                    cos_cost = self._cos_cost_simple(feat_for_bank, proto_vec)
                                    safe_to_push = (cos_cost <= float(getattr(self, "long_update_cos_thresh", 0.15)))
                                    # safe_to_push = True

                                if safe_to_push and (not hasattr(self, "_idswap_locked_ids") 
                                                    or int(root_trk.id) not in self._idswap_locked_ids):
                                    self._bank_add_feature(root_trk, feat_for_bank)
                    except Exception as e:
                        print(f"[LongBank] post-reactivate push failed for id={getattr(root_trk,'id',-1)}: {e}")
                    
                    # Merge fragments into canonical
                    if merge_ids:
                        for merge_id in merge_ids:
                            if merge_id in lost_map:
                                merge_track = lost_map[merge_id]
                                
                                # Merge feature histories
                                merge_long     = getattr(merge_track, "long_feat_mean", None)
                                canonical_long = getattr(root_trk, "long_feat_mean", None)
                                
                                if merge_long is not None:
                                    if canonical_long is None:
                                        root_trk.long_feat_mean = merge_long.copy()
                                    else:
                                        # Weighted average
                                        weight = 0.3
                                        root_trk.long_feat_mean = (
                                            (1 - weight) * canonical_long + weight * merge_long
                                        )
                                        # Renormalize
                                        norm = np.linalg.norm(root_trk.long_feat_mean)
                                        if norm > 1e-6:
                                            root_trk.long_feat_mean /= norm
                                
                                # Remove merged track
                                self.lost_stracks = [t for t in self.lost_stracks if t.id != merge_id]
                                del lost_map[merge_id]
                                
                                # Clean coexistence map
                                self._purge_from_coex([merge_id])
                                
                except Exception as e:
                    print(f"Error promoting pending track: {e}")
                    continue
        
        # --- 8. Update track states ---
        self._update_track_states(lost_stracks, removed_stracks)
        
        # --- 9. Prepare output ---
        outputs, logs, tracks_for_visual = self._prepare_output(
            activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )
        
        pending_tracks = self.pending_manager.pending_tracks
        
        try:
            self._update_coexistence_simple([t for t in self.active_tracks if t.state == TrackState.Tracked])
        except Exception as e:
            print(f"Error updating coexistence: {e}")
        finally:
            # bỏ khóa sau frame
            if hasattr(self, "_idswap_locked_ids"):
                self._idswap_locked_ids.clear()
        return outputs, logs, tracks_for_visual, pending_tracks

    def _split_detections(self, dets, embs):
        """Split detections into high and low confidence tiers."""
        dets  = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        
        second_mask = np.logical_and(
            confs > self.track_low_thresh, confs < self.track_high_thresh
        )
        
        dets_second = dets[second_mask]
        first_mask  = confs >= self.track_high_thresh
        dets_first  = dets[first_mask]
        embs_first  = embs[first_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        """Create detection objects from boxes and features."""
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [
                    STrack(det, f, max_obs=self.max_obs)
                    for (det, f) in zip(dets_first, features_high)
                ]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for det in dets_first]
        else:
            detections = []
            
        return detections

    def _separate_tracks(self):
        """Separate tracks into confirmed and unconfirmed."""
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _first_association(
        self,
        dets,
        dets_first,
        active_tracks,
        unconfirmed,
        img,
        detections,
        activated_stracks,
        refind_stracks,
        strack_pool,
    ):
        # Reset logs
        for track in strack_pool:
            track.iou_cost   = float("nan")
            track.reid_cost  = float("nan")
            track.final_cost = float("nan")
        
        if not detections or not strack_pool:
            return [], list(range(len(strack_pool))), list(range(len(detections)))
        
        # Motion prediction and camera compensation
        STrack.multi_predict(strack_pool)
        warp = self.cmc.apply(img, dets[:, :4])
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # IoU distance
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)
        iou_cost = np.clip(ious_dists.astype(np.float32), 0.0, 1.0)

        # ReID cost (short-term)
        if self.with_reid and len(strack_pool) > 0 and len(detections) > 0:
            reid_cost = embedding_distance_botsort(strack_pool, detections) / 2.0
            reid_cost[reid_cost > self.appearance_thresh] = 1.0
        else:
            reid_cost = np.ones_like(iou_cost, dtype=np.float32)
            
        # Long-term cost    
        if self.with_reid and len(strack_pool) > 0 and len(detections) > 0:
            long_cost = self._compute_long_cost(strack_pool, detections)
            long_cost = np.clip(long_cost.astype(np.float32), 0.0, 1.0)
        else:
            long_cost = np.ones_like(iou_cost, dtype=np.float32)
            
        # Apply gates
        reid_cost[ious_dists_mask] = 1.0
        long_cost[ious_dists_mask] = 1.0

        # Handle NaN values
        iou_cost  = np.nan_to_num(iou_cost , nan=1.0, posinf=1.0, neginf=1.0)
        reid_cost = np.nan_to_num(reid_cost, nan=1.0, posinf=1.0, neginf=1.0)
        long_cost = np.nan_to_num(long_cost, nan=1.0, posinf=1.0, neginf=1.0)
        
        # Final cost: motion + short ReID
        C = np.fmin(iou_cost, reid_cost).astype(np.float32)
        C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0)

        # Assignment
        matches, u_track, u_detection = linear_assignment(C, thresh=self.match_thresh)

        # Update matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det   = detections[idet]

            track.iou_cost       = float(iou_cost[itracked, idet])
            track.reid_cost      = float(reid_cost[itracked, idet]) if self.with_reid else float("nan")
            track.long_reid_cost = float(long_cost[itracked, idet])
            track.final_cost     = float(C[itracked, idet])
                 
            # Build context occlusion: active + pending (đã predict), lọc theo class nếu cần
            cls_ref = getattr(track, "cls", None) if self.per_class else None
            other_tracks_for_occlusion = self._occlusion_context(
                exclude_id=getattr(track, "id", None),
                cls_ref=cls_ref,
                predict_pending=True,
            )
            
            # Check update permissions
            allow_short, allow_long, occluded = self.allow_update(
                long_cost[itracked, idet] if self.with_reid else None,
                track,
                other_tracks=other_tracks_for_occlusion
            )
            
            track._is_occluded = bool(occluded)
            
            if track.state == TrackState.Tracked:
                track.update(
                    det, self.frame_count, img,
                    allow_feat=allow_short,     
                    allow_long=allow_long
                )
                activated_stracks.append(track)
            else:
                track.re_activate(
                    det, self.frame_count, new_id=False, img=img,
                    allow_feat=allow_short,
                    allow_long=allow_long
                )
                refind_stracks.append(track)

            if allow_long and self.with_reid:
                if int(getattr(track, "id", -1)) not in getattr(self, "_idswap_locked_ids", set()):
                    src_feat = first_not_none(getattr(det, "curr_feat", None), getattr(track, "curr_feat", None))
                    self._bank_add_feature(track, src_feat)
            
        return matches, u_track, u_detection

    def _second_association(
        self,
        dets_second,
        activated_stracks,
        lost_stracks,
        refind_stracks,
        u_track_first,
        strack_pool,
        img,
        features_second=None,
    ):
        """Perform second-tier association with low confidence detections."""
        if len(dets_second) == 0:
            # Mark remaining tracks as lost
            for i in u_track_first:
                track = strack_pool[i]
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost_stracks.append(track)
            return [], u_track_first, []

        # Create second tier detections
        has_feats = (
            self.with_reid
            and features_second is not None
            and len(features_second) == len(dets_second)
            and len(features_second) > 0
        )
        if has_feats:
            detections_second = [
                STrack(det, f, max_obs=self.max_obs)
                for det, f in zip(dets_second, features_second)
            ]
        else:
            detections_second = [STrack(det, max_obs=self.max_obs) for det in dets_second]

        # Get unmatched tracked stracks from first round
        r_tracked_stracks = [
            strack_pool[i] for i in u_track_first if strack_pool[i].state == TrackState.Tracked
        ]

        if not r_tracked_stracks:
            return [], u_track_first, list(range(len(detections_second)))

        # IoU-only matching for second tier
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)
        
        # Compute ReID costs for logging
        reid_cost_mat = None
        if (
            self.with_reid
            and len(r_tracked_stracks) > 0
            and len(detections_second) > 0
            and any(getattr(d, "curr_feat", None) is not None for d in detections_second)
        ):
            reid_cost_mat = embedding_distance_botsort(r_tracked_stracks, detections_second) / 2.0
            reid_cost_mat[reid_cost_mat > self.appearance_thresh] = 1.0
            reid_cost_mat = np.nan_to_num(reid_cost_mat, nan=1.0, posinf=1.0, neginf=1.0)

        long_cost_mat = None
        if self.with_reid and len(r_tracked_stracks) > 0 and len(detections_second) > 0:
            long_cost_mat = self._compute_long_cost(r_tracked_stracks, detections_second)
            long_cost_mat = np.nan_to_num(long_cost_mat, nan=1.0, posinf=1.0, neginf=1.0)
            bad_long = long_cost_mat > self.appearance_thresh
            long_cost_mat[bad_long] = 1.0
            if reid_cost_mat is not None:
                reid_cost_mat[bad_long] = 1.0

        # Update matched tracks (motion only, no feature updates)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]

            # Log costs
            track.iou_cost       = float(dists[itracked, idet])
            track.final_cost     = track.iou_cost
            track.reid_cost      = float(reid_cost_mat[itracked, idet]) if reid_cost_mat is not None else 1.0
            track.long_reid_cost = float(long_cost_mat[itracked, idet]) if long_cost_mat is not None else float("nan")

            # NEW: tính occlusion và gắn cờ để pause suspect/window ở tầng voting
            cls_ref = getattr(track, "cls", None) if self.per_class else None
            other_tracks_for_occlusion = self._occlusion_context(
                exclude_id=getattr(track, "id", None),
                cls_ref=cls_ref,
                predict_pending=True,
            )
            _, _, occluded2 = self.allow_update(
                long_val=None,              # vòng 2 không cần long_val
                track=track,
                other_tracks=other_tracks_for_occlusion,
            )
            track._is_occluded = bool(occluded2)

            # Update motion only (no features)
            track.frame_id = self.frame_count
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count, img=None, allow_feat=False, allow_long=False)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False, img=None, allow_feat=False, allow_long=False)
                refind_stracks.append(track)
                
        # Mark unmatched tracks as lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        return matches, u_track, u_detection

    def _update_track_states(self, lost_stracks, removed_stracks):
        """Remove tracks that have been lost for too long."""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(
        self, activated_stracks, refind_stracks, lost_stracks, removed_stracks
    ):
        """Prepare final tracking outputs."""
        # Update active tracks
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        
        
        # --- dedup active by id: lấy track có frame_id mới nhất
        uniq = {}
        for t in (self.active_tracks or []):
            tid = int(getattr(t, "id", -1))
            if tid < 0:
                continue
            prev = uniq.get(tid)
            if (prev is None) or (getattr(t, "frame_id", -1) >= getattr(prev, "frame_id", -1)):
                uniq[tid] = t
        self.active_tracks = list(uniq.values())

        # --- loại lost có id đã active
        active_ids = {int(getattr(t, "id", -1)) for t in self.active_tracks if getattr(t, "id", None) is not None}
        if active_ids:
            self.lost_stracks[:] = [t for t in self.lost_stracks if int(getattr(t, "id", -1)) not in active_ids]
        
        # Update lost tracks
        self.lost_stracks  = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)

        # Handle removed tracks
        self.removed_stracks.extend(removed_stracks)
        try:
            to_purge = {int(t.id) for t in self.removed_stracks if getattr(t, "id", None) is not None}
            self._purge_from_coex(to_purge)
        except Exception as e:
            print(f"Error purging removed tracks: {e}")
            
        # --- delete vectors in bank for removed tracks (ADD) ---
        if self.long_bank is not None:
            for t in removed_stracks:
                try:
                    if getattr(t, "id", None) is not None:
                        self.long_bank.delete_track(self.run_uid, int(t.id))
                        self._slot_pos.pop(int(t.id), None)
                        self._bank_proto_cache.pop(int(t.id), None)
                except Exception as e:
                    print(f"[LongBank] delete id={getattr(t,'id',-1)} failed: {e}")
        # -------------------------------------------------------
        
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        
        # Prepare output arrays
        outputs = []
        for t in self.active_tracks:
            if t.is_activated and hasattr(t, 'xyxy') and hasattr(t, 'id'):
                outputs.append([
                    *t.xyxy, t.id,
                    getattr(t, 'conf', 1.0),
                    getattr(t, 'cls', 0),
                    getattr(t, 'det_ind', -1)
                ])
        
        # Prepare logs
        logs = []
        for t in self.active_tracks:
            if t.is_activated:
                logs.append({
                    "track_id"      : getattr(t, 'id', -1),
                    "conf"          : getattr(t, "conf"          , float("nan")),    
                    "reid_cost"     : getattr(t, "reid_cost"     , float("nan")),
                    "long_reid_cost": getattr(t, "long_reid_cost", float("nan")),
                    "iou_cost"      : getattr(t, "iou_cost"      , float("nan")),
                    "final_cost"    : getattr(t, "final_cost"    , float("nan")),
                })
        
        # Prepare visual tracks
        tracks_for_visual = []
        for t in (self.active_tracks + self.lost_stracks):
            if getattr(t, 'latest_image') is not None:
                tracks_for_visual.append({"id": getattr(t, 'id', -1), "image": t.latest_image})

        return np.asarray(outputs, dtype=np.float32) if outputs else np.empty((0, 8), dtype=np.float32), logs, tracks_for_visual
                   
    def _max_overlap_ratio(self, track, other_tracks):
        """Calculate max overlap ratio between track and other tracks."""
        try:
            box1 = track.xyxy
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            max_ratio = 0.0

            for other in other_tracks:
                if getattr(other, 'id', None) == getattr(track, 'id', None):
                    continue
                if not hasattr(other, 'xyxy'):
                    continue
                
                box2 = other.xyxy
                xA = max(box1[0], box2[0])
                yA = max(box1[1], box2[1])
                xB = min(box1[2], box2[2])
                yB = min(box1[3], box2[3])
                inter_area = max(0, xB - xA) * max(0, yB - yA)
                if inter_area > 0:
                    ratio = inter_area / area1
                    if ratio > max_ratio:
                        max_ratio = ratio
            return max_ratio
        except Exception:
            return 0.0