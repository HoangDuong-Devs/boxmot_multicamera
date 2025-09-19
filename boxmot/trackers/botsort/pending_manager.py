# pending_manager.py

import numpy as np
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.utils.ops import xyxy2xywh, xywh2xyxy
from boxmot.trackers.botsort.botsort_track import crop_image
from boxmot.utils.matching import (
    linear_assignment,
)

def iou(bbox1, bbox2):
    """Compute IoU between 2 bounding boxes (xyxy format)."""
    try:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0
    except Exception:
        return 0.0

class PendingTrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()
    
    def __init__(self, det, frame_id, feat=None, promotion_deadline=15):
        """
        Initialize pending track.
        Args:
            det: [x1, y1, x2, y2, conf, cls, det_ind]
        """
        try:
            self.xywh    = xyxy2xywh(det[:4])
            self.conf    = det[4]
            self.cls     = det[5]
            self.det_ind = det[6]
        except Exception:
            # Fallback for malformed detections
            self.xywh    = np.array([0, 0, 1, 1], dtype=np.float32)
            self.conf    = 1.0
            self.cls     = 0
            self.det_ind = -1
        
        self.kalman_filter         = None
        self.mean, self.covariance = None, None
        
        self.state        = TrackState.Pending
        self.frame_id     = frame_id
        self.start_frame  = frame_id
        self.is_activated = False
        
        self.promotion_deadline          = promotion_deadline
        
        # Feature handling
        self.curr_feat   = None
        self.smooth_feat = None
        
        if feat is not None:
            try:
                feat = np.asarray(feat, dtype=np.float32, copy=False).ravel()
                if feat.size > 0 and np.isfinite(feat).all():
                    # Check if already normalized (norm ≈ 1.0)
                    norm = np.linalg.norm(feat)
                    if abs(norm - 1.0) > 0.01:
                        feat = normalize_feature_standard(feat, copy = False)
                        if feat is not None:
                            self.curr_feat = feat
                            self.smooth_feat = feat.copy()
                    else:   # Đã được normalize
                        self.curr_feat   = feat
                        self.smooth_feat = feat.copy()
            except Exception:
                self.curr_feat   = None
                self.smooth_feat = None
                
        self.id = -1  # Not assigned real ID yet
        
        self.first_image  = None
        self.latest_image = None
        
        self.cand_lost = {} # Store candidate lost tracks
        
    def activate(self, kalman_filter):
        """Activate the pending track with Kalman filter."""
        try:
            self.kalman_filter = kalman_filter
            self.mean, self.covariance = kalman_filter.initiate(self.xywh)
            self.is_activated = True
        except Exception as e:
            print(f"Error activating pending track: {e}")
            # Fallback initialization
            self.mean         = np.concatenate([self.xywh, np.zeros(4)])
            self.covariance   = np.eye(8)
            self.is_activated = True
        
    def predict(self):
        """Predict next state using Kalman filter."""
        if self.mean is None or self.kalman_filter is None:
            return
        
        try:
            mean_state = self.mean.copy()
            mean_state[6:8] = 0  # reset velocity
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        except Exception as e:
            print(f"Error predicting pending track: {e}")

    def update(self, det, frame_id, feat=None, img=None, other_tracks=None):
        """Update pending track with new detection."""
        try:
            self.frame_id = frame_id
            self.xywh = xyxy2xywh(det[:4])
            
            if self.kalman_filter is not None and self.mean is not None:
                self.mean, self.covariance = self.kalman_filter.update(
                    self.mean, self.covariance, self.xywh
                )
            
            self.conf = float(det[4]) if len(det) > 4 else 1.0
            
            # Update features
            if feat is not None:
                try:
                    feat = np.asarray(feat, dtype=np.float32).ravel()
                    norm = np.linalg.norm(feat)
                    
                    if np.isfinite(norm) and norm > 1e-6:
                        # Check if already normalized
                        if abs(norm - 1.0) > 0.01:
                            feat = feat / norm
                            
                        self.curr_feat = feat
                        if self.smooth_feat is None:
                            self.smooth_feat = feat.copy()
                        else:
                            # Smooth update
                            self.smooth_feat = 0.9 * self.smooth_feat + 0.1 * feat
                            # Re-normalize smooth_feat
                            smooth_norm = np.linalg.norm(self.smooth_feat)
                            if smooth_norm > 1e-6:
                                self.smooth_feat = self.smooth_feat / smooth_norm
                except Exception:
                    pass  # Keep existing features if update fails
                    
        except Exception as e:
            print(f"Error updating pending track: {e}")
        
    def can_promote(self, frame_id):
        """Check if track can be promoted to active."""
        return (frame_id - self.start_frame) >= self.promotion_deadline
    
    @property
    def xyxy(self):
        """Convert to xyxy format."""
        try:
            ret = self.mean[:4].copy() if self.mean is not None else self.xywh.copy()
            return xywh2xyxy(ret)
        except Exception:
            return np.array([0, 0, 1, 1], dtype=np.float32)

class PendingManager:
    def __init__(self, kalman_filter, promotion_deadline=15,
                 iou_thresh=0.7, appearance_thresh=0.1, match_thresh=0.8, use_dynamic_weights=True,
                w_motion_base=0.6, w_motion_start_dw=0.6, dw_start_frames=20, dw_step_frames=20,
                dw_step_delta=0.05, reid_dim=None, dw_app_split=0.5, promote_min_frames_for_lost=5, 
                proto_provider=None, vectors_provider=None, long_bank_topk=5, debug_pending_lost=False,
                qdrant_group_provider=None):
        self.pending_tracks = []
        self.kalman_filter  = kalman_filter
        self.promotion_deadline = promotion_deadline
        self.iou_thresh         = iou_thresh
        self.appearance_thresh  = appearance_thresh
        self.match_thresh       = match_thresh
        
        # Dynamic weights parameters
        self.use_dynamic_weights = use_dynamic_weights
        self.w_motion_base       = w_motion_base
        self.w_motion_start_dw   = w_motion_start_dw
        self.dw_start_frames     = dw_start_frames
        self.dw_step_frames      = dw_step_frames
        self.dw_step_delta       = dw_step_delta
        self.reid_dim            = int(reid_dim) if reid_dim is not None else None
        self.dw_app_split        = float(dw_app_split)
        self.promote_min_frames_for_lost = int(promote_min_frames_for_lost)
        self.proto_provider      = proto_provider
        self.vectors_provider    = vectors_provider
        self.long_bank_topk      = int(long_bank_topk)
        
        self.debug_pending_lost  = bool(debug_pending_lost)
        self.debug_pending_lost  = True
        
        # New: Use qdrant server
        self.qdrant_group_provider = qdrant_group_provider 
        self.run_uid               = None  # sẽ được gán từ BotSort
        
        self._norm_cache = {"frame": -1, "store": {}}
        
    # ============ LOG HELPERS ============
    def _log_pl(self, *args, **kwargs):
        """Internal logger for pending-related steps."""
        if self.debug_pending_lost:
            try:
                print(*args, **kwargs)
            except Exception:
                pass

    def _log_assignment_results(self, tag, matches, C, frame_id):
        """Pretty-print assignment results."""
        try:
            M, N = C.shape if hasattr(C, "shape") else (0, 0)

            # Normalize matches to list of tuples
            if matches is None:
                n_matches = 0
                pairs = []
            elif isinstance(matches, np.ndarray):
                n_matches = int(matches.shape[0])
                pairs = matches.tolist()
            else:
                n_matches = len(matches)
                pairs = list(matches)

            self._log_pl(f"[{tag}][frame={frame_id}] ASSIGNMENT RESULTS: MxN={M}x{N}, matches={n_matches}")
            if n_matches > 0 and M > 0 and N > 0:
                vals = [float(C[i, j]) for (i, j) in pairs]
                self._log_pl(f"  cost: min={min(vals):.3f}, max={max(vals):.3f}, avg={np.mean(vals):.3f}")
                for k, (i, j) in enumerate(pairs):
                    self._log_pl(f"    {k}: row[{i}] -> col[{j}] | cost={C[i,j]:.3f}")
            else:
                try:
                    gmin = float(np.min(C)) if hasattr(C, "size") and C.size > 0 else 1.0
                except Exception:
                    gmin = 1.0
                self._log_pl(f"  no matches; global min cost={gmin:.3f} vs thr={self.match_thresh:.3f}")
        except Exception as e:
            print(f"Error logging assignment results: {e}")
        
    # =====================================
    
    def add_pending(self, detections, frame_id, feats=None, img=None):
        """Add new pending tracks from detections."""
        if not detections:
            return
        
        for i, det in enumerate(detections):
            try:
                # feats đã được normalize từ BoTSORT main flow
                feat  = feats[i] if feats is not None else None
                track = PendingTrack(
                    det, frame_id, feat=feat,
                    promotion_deadline=self.promotion_deadline
                )
                track.activate(self.kalman_filter)
                
                if img is not None:
                    try:
                        track.latest_image = crop_image(img, det[:4])
                    except Exception:
                        pass    # Continue without image if cropping fails
                self.pending_tracks.append(track)
            except Exception as e:
                print(f"Error adding pending track {i}: {e}")
                continue
    
    def _iou_cost(self, A, B):
        if A.size == 0 or B.size == 0:
            return np.ones((len(A), len(B)), dtype=np.float32)
        C = 1.0 - np.clip(self._batch_iou(A, B), 0.0, 1.0)
        return np.clip(np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    
    def _long_cost_topk_pending_lost(self, pend_feat_list, lost_tracks, feat_dim):
        M, N = len(pend_feat_list), len(lost_tracks)
        if M == 0 or N == 0:
            return np.ones((M, N), dtype=np.float32)

        # 1) Chuẩn hoá pending feats + mask invalid
        P, p_mask = self._stack_and_l2norm_with_mask_cached(pend_feat_list, feat_dim, tag="pend")

        # 2) SERVER-SIDE (nếu bật & có provider)
        if self.qdrant_group_provider is not None:
            valid_rows = np.where(~p_mask)[0]
            if len(valid_rows) > 0:
                queries = P[valid_rows]                 # (m_valid, D)
                lost_ids = [int(getattr(t, "id", -1)) for t in lost_tracks]
                run_uid  = self.run_uid
                resp = self.qdrant_group_provider(run_uid, lost_ids, queries, k=self.long_bank_topk)
                if resp is not None:
                    C = np.ones((M, N), dtype=np.float32)
                    for r_idx, dmap in zip(valid_rows, resp):
                        for j, lost in enumerate(lost_tracks):
                            tid = int(getattr(lost, "id", -1))
                            sim = dmap.get(tid, None)
                            if sim is None:
                                C[r_idx, j] = 1.0
                            else:
                                C[r_idx, j] = 0.5 * (1.0 - float(sim))
                    return np.clip(C, 0.0, 1.0)

        # 3) FALLBACK client-side (giữ nguyên logic cũ)
        return self._long_cost_topk_pending_lost_client(P, p_mask, lost_tracks, feat_dim)

    # --- ADD: tách thân client-side cũ vào hàm riêng cho rõ ràng ---
    def _long_cost_topk_pending_lost_client(self, P, p_mask, lost_tracks, feat_dim):
        M, N = P.shape[0], len(lost_tracks)
        if N == 0:
            return np.ones((M, 0), dtype=np.float32)
        C = np.ones((M, N), dtype=np.float32)
        k_req = max(1, int(self.long_bank_topk))
        for j, lost in enumerate(lost_tracks):
            V = None
            if self.vectors_provider is not None:
                try:
                    V = self.vectors_provider(lost)
                except Exception:
                    V = None
            if V is None:
                v = None
                if self.proto_provider is not None:
                    try:
                        v, ok = self.proto_provider(lost)
                        v = v if ok else None
                    except Exception:
                        v = None
                if v is not None:
                    v = np.asarray(v, dtype=np.float32).ravel()
                    n = np.linalg.norm(v)
                    if np.isfinite(n) and n > 1e-6 and abs(n - 1.0) > 0.01:
                        v = v / n
                    V = v[None, :] if v is not None else None

            if V is None:
                C[:, j] = 1.0
                continue

            V = np.asarray(V, dtype=np.float32)
            if V.ndim == 1: V = V[None, :]
            vn = np.linalg.norm(V, axis=1, keepdims=True)
            need = np.abs(vn - 1.0) > 1e-2
            if np.any(need):
                V = V / np.clip(vn, 1e-6, None)

            try:
                sims = np.clip(P @ V.T, -1.0, 1.0)        # (M, n_j)
                k_use = min(k_req, sims.shape[1])
                if k_use <= 0:
                    C[:, j] = 1.0
                    continue
                topk = np.sort(sims, axis=1)[:, -k_use:]  # (M, k)
                mean_sims = topk.mean(axis=1)             # (M,)
                col = 0.5 * (1.0 - mean_sims)
                col = np.where(p_mask, 1.0, col).astype(np.float32, copy=False)
                C[:, j] = np.clip(col, 0.0, 1.0)
            except Exception:
                C[:, j] = 1.0
        return C

    
    def _stack_and_l2norm_with_mask(self, feature_list, dim):
        """Optimized feature stacking and normalization."""
        k = len(feature_list)
        mat = np.zeros((k, dim), dtype=np.float32)
        mask = np.zeros(k, dtype=bool)
        
        # Pre-allocate norm array
        norms = np.zeros(k, dtype=np.float32)
        
        # Single pass through features
        for i, f in enumerate(feature_list):
            if f is None:
                mask[i] = True
                continue
                
            try:
                v = np.asarray(f, dtype=np.float32).ravel()
                if v.shape[0] != dim:
                    mask[i] = True
                    continue
                    
                norms[i] = np.linalg.norm(v)
                if not np.isfinite(norms[i]) or norms[i] < 1e-6:
                    mask[i] = True
                    continue
                    
                # Only normalize if needed
                if abs(norms[i] - 1.0) > 0.01:
                    mat[i] = v / norms[i]
                else:
                    mat[i] = v
            except Exception:
                mask[i] = True
        
        return mat, mask

    def _stack_and_l2norm_with_mask_cached(self, feature_list, dim, tag):
        """Cache with optimized key generation."""
        # Use hash for faster key comparison
        key_data = (tag, dim, len(feature_list))
        for f in feature_list:
            if f is None:
                key_data += (None,)
            else:
                key_data += (id(f), getattr(f, 'shape', None))
        
        key = hash(key_data)
        store = self._norm_cache["store"]
        
        if key in store:
            return store[key]
            
        mat, mask = self._stack_and_l2norm_with_mask(feature_list, dim)
        store[key] = (mat, mask)
        return mat, mask

    def update_pending(self, detections, feats, lost_tracks, frame_id,
                      with_reid=True, img=None, other_tracks=None):
        """
        Update pending tracks with detections and lost tracks.
        Returns: unmatched_dets, unmatched_feats, merged_to_lost
        """
        if self._norm_cache["frame"] != frame_id:
            self._norm_cache["frame"] = frame_id
            self._norm_cache["store"] = {}
        
        unmatched_dets = list(detections) if detections else []
        unmatched_feats = list(feats) if feats is not None else [None] * len(unmatched_dets)
        merged_to_lost = []

        if not self.pending_tracks:
            return unmatched_dets, unmatched_feats, merged_to_lost

        # Predict all pending tracks
        for t in self.pending_tracks:
            t.predict()

        keep_idx = set()
        feat_dim = self.reid_dim

        if self.pending_tracks:
            pending_boxes = np.asarray([p.xyxy for p in self.pending_tracks], dtype=np.float32)
        else:
            pending_boxes = np.empty((0, 4), dtype=np.float32)
        
        # 1) Pending vs Detections matching
        if unmatched_dets:
            try:
                det_boxes = np.array([d[:4] for d in unmatched_dets], dtype=np.float32)

                iou_cost = self._iou_cost(pending_boxes, det_boxes)

                if with_reid:
                    # Prefer smooth_feat, fallback to curr_feat
                    pend_feat_list = [
                        (p.smooth_feat if getattr(p, "smooth_feat", None) is not None else p.curr_feat)
                        for p in self.pending_tracks
                    ]
                    
                    pending_feats, p_mask = self._stack_and_l2norm_with_mask_cached(pend_feat_list, feat_dim, tag="pend")
                    det_feats, d_mask     = self._stack_and_l2norm_with_mask_cached(unmatched_feats,  feat_dim, tag="det_u")

                    cos = np.clip(pending_feats @ det_feats.T, -1.0, 1.0)
                    emb_cost = 0.5 * (1.0 - cos)

                    invalid_pairs = p_mask[:, None] | d_mask[None, :]
                    emb_cost[invalid_pairs] = 1.0

                    valid_pairs = ~invalid_pairs
                    emb_cost[valid_pairs & (emb_cost > self.appearance_thresh)] = 1.0

                    iou_gate = iou_cost > float(self.iou_thresh)
                    emb_cost[iou_gate] = 1.0
                    cost_matrix = np.fmin(iou_cost, emb_cost).astype(np.float32)
                else:
                    cost_matrix = iou_cost.astype(np.float32)

                cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=1.0)
                
                matches, _, u_det_idx = linear_assignment(cost_matrix, thresh=self.match_thresh)

                for ipend, idet in matches:
                    try:
                        det = unmatched_dets[idet]
                        feat = unmatched_feats[idet]

                        # Filter other tracks to avoid self-reference
                        other_tracks_filtered = None
                        if other_tracks:
                            other_tracks_filtered = [
                                t for t in other_tracks
                                if not (hasattr(t, "xyxy") and np.allclose(t.xyxy, det[:4], atol=1e-3))
                            ]

                        self.pending_tracks[ipend].update(det, frame_id, feat, img=img, 
                                                        other_tracks=other_tracks_filtered)
                        keep_idx.add(ipend)

                        # Update images
                        if img is not None:
                            try:
                                cropped_img = crop_image(img, det[:4])
                                self.pending_tracks[ipend].latest_image = cropped_img
                                if self.pending_tracks[ipend].first_image is None:
                                    self.pending_tracks[ipend].first_image = cropped_img.copy()
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Error updating pending track {ipend}: {e}")
                        continue

                # Keep unmatched detections and features
                unmatched_dets  = [unmatched_dets[i] for i in u_det_idx]
                unmatched_feats = [unmatched_feats[i] for i in u_det_idx]

            except Exception as e:
                print(f"Error in pending-detection matching: {e}")

        # 2) Pending vs Lost Tracks matching (accumulation only)
        if lost_tracks:
            try:
                lost_boxes    = np.array([t.xyxy for t in lost_tracks], dtype=np.float32)
                
                iou_cost = self._iou_cost(pending_boxes, lost_boxes)
                
                if with_reid:
                    # Pending feature (ưu tiên smooth)
                    pend_feat_list = [
                        (p.smooth_feat if p.smooth_feat is not None else p.curr_feat)
                        for p in self.pending_tracks
                    ]
                    # Long-term top-k pooling cost
                    cost_l = self._long_cost_topk_pending_lost(pend_feat_list, lost_tracks, feat_dim)

                    # Gate theo appearance_thresh (mặc định 0.1)
                    bad_l = (cost_l > float(self.appearance_thresh))
                    cost_l = np.where(bad_l, 1.0, cost_l).astype(np.float32, copy=False)
                else:
                    cost_l = np.ones_like(iou_cost, dtype=np.float32)

                cost_l   = np.nan_to_num(cost_l, nan=1.0, posinf=1.0, neginf=1.0)
                iou_cost = np.nan_to_num(iou_cost, nan=1.0, posinf=1.0, neginf=1.0)

                # Dynamic weights (giữ nguyên logic cũ)
                use_dw      = bool(self.use_dynamic_weights)
                base_motion = self.w_motion_start_dw if use_dw else self.w_motion_base

                M, N = iou_cost.shape
                Wm = np.full((M, N), base_motion, dtype=np.float32)
                if use_dw:
                    for j, lost in enumerate(lost_tracks):
                        if lost.state == TrackState.Lost:
                            end_f = getattr(lost, "end_frame", getattr(lost, "frame_id", None))
                            lost_f = max(0, frame_id - end_f) if end_f is not None else 0
                            if lost_f >= int(self.dw_start_frames):
                                steps = 1 + (lost_f - int(self.dw_start_frames)) // max(1, int(self.dw_step_frames))
                                val = max(0.0, base_motion - steps * float(self.dw_step_delta))
                                Wm[:, j] = val

                Wa = 1.0 - Wm
                cost_matrix = (Wm * iou_cost + Wa * cost_l).astype(np.float32)
                cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=1.0)

                # --- LOG ma trận trước khi accumulate
                if self.debug_pending_lost:
                    self._log_pl(f"[PL][frame={frame_id}] cost_matrix shape={cost_matrix.shape}, "
                                 f"thr={self.match_thresh:.3f}")
                    self._log_pl(f"  IoU cost stats: min={float(np.min(iou_cost)):.3f}, "
                                 f"max={float(np.max(iou_cost)):.3f}")
                    self._log_pl(f"  ReID cost stats: min={float(np.min(cost_l)):.3f}, "
                                 f"max={float(np.max(cost_l)):.3f}")
                    self._log_pl(f"  Mix cost stats:  min={float(np.min(cost_matrix)):.3f}, "
                                 f"max={float(np.max(cost_matrix)):.3f}")
                    
                # Nhiều-nhiều (accumulation)
                thr = float(self.match_thresh)
                pairs = [(ip, jl)
                         for ip in range(M)
                         for jl in range(N)
                         if cost_matrix[ip, jl] <= thr]

                # --- LOG danh sách cặp tích lũy ---
                self._log_assignment_results("PL", pairs, cost_matrix, frame_id)
                
                if self.debug_pending_lost:
                    try:
                        # gom các cặp theo pending index
                        by_pend = {}
                        for (ip, jl) in pairs:
                            lost_id = getattr(lost_tracks[jl], "id", None)
                            by_pend.setdefault(ip, []).append((jl, lost_id))

                        # in cho từng pending
                        for ip, items in by_pend.items():
                            p = self.pending_tracks[ip]
                            pid = getattr(p, "det_ind", -1)
                            ptag = f"pend[{ip}] det_ind={pid}"

                            details = []
                            for jl, lost_id in items:
                                mix = float(cost_matrix[ip, jl])
                                iou = float(iou_cost[ip, jl])
                                # cost_l là ReID long-topk khi with_reid=True, nếu không thì NaN
                                reid = float(cost_l[ip, jl]) if with_reid else float("nan")
                                wm = float(Wm[ip, jl])
                                wa = float(Wa[ip, jl])
                                details.append(
                                    f"lost_id={lost_id} mix={mix:.3f} (IoU={iou:.3f}, ReID={reid:.3f}, Wm={wm:.2f}, Wa={wa:.2f})"
                                )

                            if details:
                                self._log_pl(f"[PL][frame={frame_id}] {ptag} -> " + "; ".join(details))
                            else:
                                self._log_pl(f"[PL][frame={frame_id}] {ptag} -> (no matches ≤ thr)")
                    except Exception as e:
                        print(f"Error logging per-pending matches: {e}")

                for ipend, ilost in pairs:
                    try:
                        pending = self.pending_tracks[ipend]
                        lost    = lost_tracks[ilost]
                        
                        # dùng MIX COST để ghi nhận (đã qua thr vì nằm trong 'pairs')
                        mix_cost = float(cost_matrix[ipend, ilost])
                        
                        st = pending.cand_lost.get(lost.id)
                        if st is None:
                            st = {"frames": 0, "last_seen": frame_id, "best": 1.0, "avg": 0.0}
                            pending.cand_lost[lost.id] = st
                            
                        st["frames"]    += 1
                        st["last_seen"]  = frame_id
                        # cập nhật best/avg để debug dùng về sau
                        st["best"] = min(st["best"], mix_cost)
                        st["avg"]  = st["avg"] + (mix_cost - st["avg"]) / st["frames"]
                    except Exception as e:
                        print(f"Error accumulating lost match: {e}")
                        continue

            except Exception as e:
                print(f"Error in pending-lost matching: {e}")

        # 3) Cleanup - keep only matched pending tracks
        try:
            self.pending_tracks = [
                t for i, t in enumerate(self.pending_tracks)
                if i in keep_idx
            ]
        except Exception as e:
            print(f"Error cleaning up pending tracks: {e}")

        return unmatched_dets, unmatched_feats, []

    def promote_pending(self, frame_id):
        """Promote eligible pending tracks."""
        try:
            promotables = [t for t in self.pending_tracks if t.can_promote(frame_id)]
            if self.debug_pending_lost and promotables:
                for idx, p in enumerate(promotables):
                    waited = frame_id - getattr(p, "start_frame", frame_id)
                    detind = getattr(p, "det_ind", -1)
                    feats  = "Y" if (getattr(p, "smooth_feat", None) is not None or getattr(p, "curr_feat", None) is not None) else "N"
                    self._log_pl(
                        f"[PROMOTE][frame={frame_id}] pend[{idx}] det_ind={detind} "
                        f"waited={waited} thr={self.promotion_deadline} "
                        f"cand_lost=({self._summarize_cand_lost(p)}) feat={feats}"
                    )
            self.pending_tracks = [t for t in self.pending_tracks if t not in promotables]
            return promotables
        except Exception as e:
            print(f"Error promoting pending tracks: {e}")
            return []

    def cleanup_expired(self, frame_id):
        """Remove expired pending tracks."""
        try:
            expired_tracks = [t for t in self.pending_tracks if t.can_promote(frame_id)]
            self.pending_tracks = [t for t in self.pending_tracks if t not in expired_tracks]
        except Exception as e:
            print(f"Error cleaning expired tracks: {e}")

    def _batch_iou(self, boxes1, boxes2):
        """Optimized batch IoU computation."""
        try:
            boxes1 = boxes1.astype(np.float32, copy=False)
            boxes2 = boxes2.astype(np.float32, copy=False)
            
            # Pre-compute areas once
            area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
            area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
            
            # Vectorized intersection calculation
            x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
            y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
            x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
            y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
            
            # Combined clipping and multiplication
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            union = area1[:, None] + area2[None, :] - inter
            
            return np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-6)
        except Exception as e:
            print(f"Error computing batch IoU: {e}")
            return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        
    def resolve_id_fragments(self, cand_ids, lost_map, pending_track, frame_id):
        """
        Quyết định canonical/merge chỉ theo điều kiện: 
        - mỗi ID phải đạt >= need_frames trong cand_lost của pending
        - canonical = ID nhỏ nhất
        - merge_ids = còn lại
        (KHÔNG xử lý coexistence ở đây; BotSort sẽ dùng coex_map để chặn sau)
        """
        try:
            if len(cand_ids) == 1:
                return cand_ids[0], []

            need_frames = int(getattr(self, "promote_min_frames_for_lost", 5))
            elig = []
            for lid in cand_ids:
                st = (pending_track.cand_lost or {}).get(lid)
                if st and int(st.get("frames", 0)) >= need_frames and lid in lost_map:
                    elig.append(lid)

            if not elig:
                return None, []

            canonical_id = min(elig)
            merge_ids = [lid for lid in elig if lid != canonical_id]
            return canonical_id, merge_ids
        except Exception as e:
            print(f"Error resolving ID fragments (min-ID only): {e}")
            return None, []
        
    def _summarize_cand_lost(self, p):
        try:
            cl = getattr(p, "cand_lost", {}) or {}
            return ", ".join([f"id={lid}(fr={int(st.get('frames',0))})" for lid, st in cl.items()]) or "-"
        except Exception:
            return "?"
        
def normalize_feature_standard(feat, eps=1e-6, copy=True):
    if feat is None: return None
    v = np.asarray(feat, np.float32, copy=copy).ravel()
    if v.size==0 or not np.isfinite(v).all(): return None
    n = np.linalg.norm(v);  return None if not np.isfinite(n) or n<eps else (v/n)
