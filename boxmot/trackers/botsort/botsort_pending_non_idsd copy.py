# botsort_pending_non_idsd.py

from pathlib import Path
import numpy as np
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
from boxmot.trackers.botsort.botsort_utils import OptimizedOverlapCalculator

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
        use_qdrant          : bool = True,
        qdrant_host         : str = "localhost",
        qdrant_port         : int = 6333,
        qdrant_collection   : str ="long_term_reid",
        bank_slots          : int = 20,
        
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
        
        self.occlusion_overlap_thresh = 0.45 # Diện tích bị che -> không cập nhật
        self.long_update_cos_thresh   = 0.12 # Feature drift    -> không cập nhật
        self.max_obs = 50
        
        reid_dim = getattr(self.model, "feature_dim", 512) if self.with_reid and self.model else None
        self.long_bank_topk = 5   # số điểm trong bank để pooling
                                
        # --- Long Feature Bank (REPLACE) ---
        import uuid as _uuid
        self.run_uid    = _uuid.uuid4().hex[:8]
        self.bank_slots = int(bank_slots)

        # Ring-slot cho từng track id
        self._slot_pos           = {}
        # Centroid cache (TTL=5f): giảm fetch/tính mean; clear khi add new feat.
        self._bank_proto_cache   = {} 
        self._bank_proto_ttl     = 5 

        self.overlap_calc = OptimizedOverlapCalculator()
        
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


        self.coex_map = {}   # id -> set các id đã từng cùng xuất hiện (active_track và lost_track)              

        qprov = (lambda run_uid, lost_ids, queries, k:
                self.long_bank.grouped_topk_mean(run_uid, lost_ids, queries, k)
                ) if (self.long_bank and self.long_bank.backend_name == "qdrant") else None
              
        self.pending_manager = PendingManager(
            kalman_filter              =self.kalman_filter,
            promotion_deadline         =15,
            iou_thresh                 =0.7,
            appearance_thresh          =0.10,
            match_thresh               =0.7,
            use_dynamic_weights        =self.use_dynamic_weights,
            w_motion_base              =self.w_motion_base,
            w_motion_start_dw          =self.w_motion_start_dw,
            dw_start_frames            =self.dw_start_frames,
            dw_step_frames             =self.dw_step_frames,
            dw_step_delta              =self.dw_step_delta,
            reid_dim                   =reid_dim,
            promote_min_frames_for_lost=5,
            proto_provider             =self._bank_centroid_for_track,
            vectors_provider           =self._bank_vectors_for_track,
            long_bank_topk             =self.long_bank_topk,
            debug_pending_lost         =True,
            qdrant_group_provider      =qprov,  # không cần truyền use_qdrant_server  
        )        
        
        self.pending_manager.run_uid = self.run_uid
        
        self._frame_cache_bank = {} # Vectors cache (1–2 frame): tránh query Qdrant lặp trong cùng frame.
        
        # Top-k vectors cache (TTL=3f): tránh cắt/lấy lại nhiều lần trong matching.
        self._topk_cache = {}
        self._cache_ttl  = 4 
        
        self.debug_long = True  # bật debug
        
        # -----------------------------------
            
    # ------------- Long Bank Helpers ---------------
    def _bank_next_slot(self, tid: int) -> int:
        pos = self._slot_pos.get(int(tid), 0)
        self._slot_pos[int(tid)] = (pos + 1) % self.bank_slots
        return pos
        
    def _bank_add_feature(self, track: STrack, feat: np.ndarray):
        """
        Add a feature to the long-term bank with safety checks:
        - validate & L2-norm feat
        - compare to current bank centroid (if any)
        - skip push if cosine distance > self.long_update_cos_thresh
        - ring-slot upsert + proto cache invalidate
        """
        if self.long_bank is None or (not self.with_reid) or feat is None:
            return
        try:
            v = np.asarray(feat, dtype=np.float32).ravel()
            if v.size == 0 or not np.isfinite(v).all():
                return

            # L2-normalize (backends also L2-normalize, nhưng normalize sớm giúp so cos chính xác)
            n = np.linalg.norm(v)
            if (not np.isfinite(n)) or n < 1e-6:
                return
            if abs(n - 1.0) > 0.01:
                v = v / n

            # (optional) reject outliers vs current centroid
            th = float(getattr(self, "long_update_cos_thresh", 0.15))  # distance in [0,1]
            try:
                proto_vec, ok = self._bank_centroid_for_track(track)
            except Exception:
                proto_vec, ok = (None, False)

            safe_to_push = True
            if ok and proto_vec is not None:
                # proto_vec đã chuẩn hoá trong _bank_centroid_for_track
                s = float(np.clip(np.dot(v, np.asarray(proto_vec, np.float32).ravel()), -1.0, 1.0))
                cos_dist = 0.5 * (1.0 - s)
                safe_to_push = (cos_dist <= th)

            if not safe_to_push:
                return

            slot = self._bank_next_slot(track.id)
            extra = {"frame_id": int(self.frame_count), "cls": int(getattr(track, "cls", 0))}
            self.long_bank.add_ring(self.run_uid, int(track.id), v, slot, extra=extra)

            # invalidate centroid cache cho track này
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
                    if np.isfinite(n) and n > 1e-6 and abs(n - 1.0) > 0.01:
                        v = v / n
                    return v.astype(np.float32), True
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

    def _build_overlap_cache_for_frame(self):
        # gom toàn bộ Tracked + Pending, không lọc theo class
        ctx_all = self._occlusion_context(
            exclude_id=None, predict_pending=False
        )
        if ctx_all:    
            self.overlap_calc.build_cache_for_context(
                ctx_all, self.frame_count, threshold=self.occlusion_overlap_thresh
            )

    def allow_update(self, long_val, track):
        long_th = float(getattr(self, "long_update_cos_thresh", 0.15))
        occ_th  = float(getattr(self, "occlusion_overlap_thresh", 0.45))

        long_ok = True if long_val is None else (long_val <= long_th)

        occluded = False  # <-- thêm dòng này
        try:
            info = self.overlap_calc.get_result(track)
            occluded = bool(info.get("is_occluded", False))
        except Exception:
            occluded = False

        allow_short = True
        allow_long  = bool(allow_short and long_ok and (not occluded))
        return allow_short, allow_long, occluded

    def _occlusion_context(self, exclude_id=None, predict_pending=False):
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

        return ctx
    
    def _update_coexistence_simple(self, tracks):
        """Optimized coexistence tracking."""
        try:
            # Filter once instead of in loop
            live = [t for t in tracks if t.state == TrackState.Tracked]
            live_ids = [int(t.id) for t in live]
            n = len(live_ids)
            
            # Pre-allocate sets
            for ida in live_ids:
                if ida not in self.coex_map:
                    self.coex_map[ida] = set()
            
            # Single loop for all pairs
            for i in range(n):
                ida = live_ids[i]
                for j in range(i + 1, n):
                    idb = live_ids[j]
                    if ida != idb:
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

    def _get_cached_topk_vectors(self, track, k: int):
        """Lấy tối đa k vector gần nhất từ bank, có cache TTL ngắn theo frame."""
        tid = int(getattr(track, 'id', -1))
        if tid < 0:
            return None

        key = (tid, self.frame_count)
        v = self._topk_cache.get(key)
        if v is not None:
            return v

        V = self._bank_vectors_for_track(track)  # đã có cache 1–2 frame ở _frame_cache_bank
        if V is None or V.size == 0:
            self._topk_cache[key] = None
            # dọn cache cũ
            old = [kk for kk in self._topk_cache if self.frame_count - kk[1] > self._cache_ttl]
            for kk in old: self._topk_cache.pop(kk, None)
            return None

        # lấy k vector cuối (coi như mới nhất)
        if V.shape[0] > k:
            V = V[-k:]
        self._topk_cache[key] = V

        # dọn cache cũ
        old = [kk for kk in self._topk_cache if self.frame_count - kk[1] > self._cache_ttl]
        for kk in old: self._topk_cache.pop(kk, None)
        return V

    def _track_det_topk_cost_row_optimized(self, V, D, dvalid, k=8):
        N = D.shape[0] if D is not None else 0
        if V is None or N == 0:
            return np.ones(N, dtype=np.float32)
        
        # Single matrix multiplication
        S = np.dot(V, D.T)
        np.clip(S, -1.0, 1.0, out=S)
        
        k_use = min(k, S.shape[0])
        if k_use <= 0:
            return np.ones(N, dtype=np.float32)
        
        # Optimized top-k selection
        if k_use < S.shape[0]:
            # Use partition instead of sort for better performance
            topk_indices = np.argpartition(S, -k_use, axis=0)[-k_use:]
            topk_values = S[topk_indices, np.arange(N)]
            sims = np.mean(topk_values, axis=0)
        else:
            sims = np.mean(S, axis=0)
        
        # Vectorized cost computation
        cost = 0.5 * (1.0 - sims)
        return np.where(dvalid, cost, 1.0).astype(np.float32, copy=False)

    def _compute_long_cost(self, tracks, dets):
        M, N = len(tracks), len(dets)
        if M == 0 or N == 0:
            return np.ones((M, N), dtype=np.float32)

        D, dvalid = build_feat_matrix(dets, ("smooth_feat", "curr_feat"))

        C = np.ones((M, N), dtype=np.float32)
        for i, t in enumerate(tracks):
            V = self._get_cached_topk_vectors(t, self.long_bank_topk)
            if V is None or V.size == 0:
                continue
            row = self._track_det_topk_cost_row_optimized(V, D, dvalid, self.long_bank_topk)
            C[i, :] = row

        return np.clip(C, 0.0, 1.0)          

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
        feats_for_pending  = []
        raw_unmatched_dets = []
        
        # From first tier (high confidence)
        for idx in u_detection_first:
            det_box = dets_first[idx]
            raw_unmatched_dets.append(det_box)
            if self.with_reid and idx < len(features_high):
                feats_for_pending.append(features_high[idx].astype(np.float32, copy=False))
            else:
                feats_for_pending.append(None)
                
        # From second tier (low confidence)
        for idx in u_detection_second:
            raw_unmatched_dets.append(dets_second[idx])
            if self.with_reid and (features_second is not None) and idx < len(features_second):
                feats_for_pending.append(features_second[idx].astype(np.float32, copy=False))
            else:
                if self.with_reid and self.model is not None and embs is None:
                    try:
                        roi = dets_second[idx, 0:4]
                        f = self.model.get_features(np.asarray([roi]), img)[0].astype(np.float32, copy=False)
                        feats_for_pending.append(f)
                    except Exception:
                        feats_for_pending.append(None)
                else:
                    feats_for_pending.append(None)

        lost_pool = joint_stracks(self.lost_stracks, lost_stracks)
        
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
        
        # --- 6.5. Cập nhật coexistence cho frame hiện tại (trước khi promote) ---
        try:
            live_now = [t for t in self.active_tracks if t.state == TrackState.Tracked]
            # thêm các track vừa khẳng định lại trong frame này
            live_now += [t for t in activated_stracks if t.state == TrackState.Tracked]
            live_now += [t for t in refind_stracks    if t.state == TrackState.Tracked]
            if live_now:
                self._update_coexistence_simple(live_now)
        except Exception as e:
            print(f"Error updating coexistence (pre-promote): {e}")
            
        # --- 7. Promote pending ---
        promotables = self.pending_manager.promote_pending(self.frame_count)

        if promotables:
            lost_map = {}
            
            for t in (self.lost_stracks + lost_stracks):
                if getattr(t, "id", None) is not None:
                    lost_map[int(t.id)] = t
                    
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
                    
                    # chỉ lọc các merge_ids bị xung đột với canonical
                    safe_merge_ids   = [mid for mid in merge_ids if mid not in coex_set]
                    conflict_ids     = [mid for mid in merge_ids if mid in coex_set]

                    # luôn re-activate canonical_id
                    root_trk = lost_map[canonical_id]
                    root_trk.re_activate(p_track, self.frame_count, new_id=False, img=img)
                    refind_stracks.append(root_trk)
                    del lost_map[canonical_id]

                    # (tuỳ chọn) log cho dễ debug
                    print(f"[PROMOTE] canonical={canonical_id}, merge={safe_merge_ids}, filtered_by_coex={conflict_ids}")

                    # --- push 1 vector vào bank như cũ ---
                    try:
                        if self.long_bank is not None and self.with_reid:
                            feat_for_bank = first_not_none(
                                getattr(root_trk, "curr_feat", None),
                                getattr(p_track, "smooth_feat", None),
                                getattr(p_track, "curr_feat", None),
                            )

                            if feat_for_bank is not None:
                                self._bank_add_feature(root_trk, feat_for_bank)
                    except Exception as e:
                        print(f"[LongBank] post-reactivate push failed for id={getattr(root_trk,'id',-1)}: {e}")

                    # chỉ merge các ID không xung đột
                    for merge_id in safe_merge_ids:
                        if merge_id in lost_map:
                            merge_track = lost_map[merge_id]
                            # merge feature histories (như code cũ)
                            merge_long     = getattr(merge_track, "long_feat_mean", None)
                            canonical_long = getattr(root_trk,  "long_feat_mean", None)
                            if merge_long is not None:
                                if canonical_long is None:
                                    root_trk.long_feat_mean = merge_long.copy()
                                else:
                                    weight = 0.3
                                    root_trk.long_feat_mean = (1 - weight) * canonical_long + weight * merge_long
                                    nrm = np.linalg.norm(root_trk.long_feat_mean)
                                    if nrm > 1e-6:
                                        root_trk.long_feat_mean /= nrm

                            self.lost_stracks = [t for t in self.lost_stracks if t.id != merge_id]
                            del lost_map[merge_id]
                            self._purge_from_coex([merge_id])
                            
                            if self.long_bank is not None:
                                try:
                                    self.long_bank.delete_track(self.run_uid, int(merge_id))
                                except Exception as e:
                                    print(f"[LongBank] delete merge id = {merge_id} failed: {e}")
                                
                            # dọn cache cục bộ
                            self._slot_pos.pop(int(merge_id), None)
                            self._bank_proto_cache.pop(int(merge_id), None)
                            self._frame_cache_bank.pop(int(merge_id), None)
                                
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

        # Build occlusion cache with latest predicted boxes (Tracked + Pending predicted)
        try:
            self._build_overlap_cache_for_frame()
        except Exception:
            pass
            
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
                 
            # Check update permissions
            allow_short, allow_long, occluded = self.allow_update(
                long_cost[itracked, idet] if self.with_reid else None,
                track,
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

            _, _, occluded2 = self.allow_update(
                long_val=None,
                track=track,
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

        self.removed_stracks.extend(removed_stracks)
        
        removed_batch = removed_stracks
        # Purge coexistence        
        try:
            to_purge = {
                int(t.id) for t in self.removed_stracks 
                if getattr(t, "id", None) is not None
            }
            if to_purge:
                self._purge_from_coex(to_purge)
        except Exception as e:
            print(f"Error purging removed tracks: {e}")
            
        # 2) Xóa vectors/caches trong LongBank cho removed IDs
        if self.long_bank is not None:
            for t in removed_stracks:
                try:
                    if getattr(t, "id", None) is not None:
                        tid = int(t.id)
                        self.long_bank.delete_track(self.run_uid, int(t.id))
                        self._slot_pos.pop(int(t.id), None)
                        self._bank_proto_cache.pop(int(t.id), None)
                        self._frame_cache_bank.pop(tid, None)
                except Exception as e:
                    print(f"[LongBank] delete id={getattr(t,'id',-1)} failed: {e}")
        
        # 3) Giải phóng ảnh trên các track vừa removed
        for t in removed_batch:
            try:
                if hasattr(t, "latest_image"): t.latest_image = None
                if hasattr(t, "first_image"):  t.first_image  = None
            except Exception:
                pass
        
        # 4) Loại lost theo batch removed
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_batch)
        
        # 5) không tích lũy lịch sử removed
        self.removed_stracks = []        
        
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
                   