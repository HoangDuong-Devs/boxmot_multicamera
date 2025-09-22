# botsort_track.py

from collections import deque
import numpy as np

from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh

class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, det, feat=None, feat_history=50, max_obs=50):
        """Initialize STrack with detection and optional feature."""
        try:
            self.xywh    = xyxy2xywh(det[:4])
            self.conf    = float(det[4]) if len(det) > 4 else 1.0
            self.cls     = int(det[5]) if len(det) > 5 else 0
            self.det_ind = int(det[6]) if len(det) > 6 else -1
        except Exception as e:
            # Fallback for malformed detections
            self.xywh    = np.array([0, 0, 1, 1], dtype=np.float32)
            self.conf    = 1.0
            self.cls     = 0
            self.det_ind = -1
            
        self.max_obs = max_obs

        self.kalman_filter         = None
        self.mean, self.covariance = None, None
        self.is_activated          = False
        self.tracklet_len          = 0

        self.cls_hist = []
        self.history_observations = deque(maxlen=self.max_obs)
        self.features             = deque(maxlen=feat_history)
        self.smooth_feat          = None
        self.curr_feat            = None
        self.update_ratio         = 0.1

        self.update_cls(self.cls, self.conf)
        self.first_feat   = None
        self.latest_image = None
        self.feat_drift_thresh = 0.1
        
        if feat is not None:
            try:
                self.update_features_split(feat, img=None, allow_long=False)
            except Exception as e:
                print(f"Error initializing features: {e}")

        # Long-term embedding buffer
        self._ensure_long_buffers()
    
    @classmethod
    def create_empty_with_new_id(cls, cls_id: int = 0, conf: float = 1.0):
        """
        Tạo 1 STrack rỗng mang ID MỚI:
        - Có mỗi id mới, xywh tối thiểu để không crash.
        - Không KF, không feature, không history.
        - Ở trạng thái Lost (để không vào pool).
        """
        det = np.array([0, 0, 1, 1, conf, cls_id, -1], dtype=np.float32)
        obj = cls(det=det, feat=None, max_obs=1)
        # cấp ID mới ngay tại đây
        obj.id = cls.next_id()

        # xoá mọi state/motion/feat
        obj.kalman_filter = None
        obj.mean, obj.covariance = None, None
        obj.is_activated = False
        obj.tracklet_len = 0

        obj.features.clear()
        obj.smooth_feat  = None
        obj.curr_feat    = None
        obj.first_feat   = None
        obj.latest_image = None

        obj._ensure_long_buffers()
        obj.long_feats.clear()
        obj.long_feat_mean = None
        obj._long_sum = None
        obj._long_len = 0

        # để ở LOST + không active
        obj.state = TrackState.Lost
        obj.just_initiated = False
        obj._is_stub = True
        return obj
    
    def _ensure_long_buffers(self):
        """Ensure long-term feature buffers are initialized."""
        if not hasattr(self, "long_feats"):
            self.long_feats     = deque(maxlen=20)
            self.long_feat_mean = None
            self._long_sum      = None
            self._long_len      = 0
            self.long_stride    = 5

    def update_features(self, feat, img=None):
        """Update both short and long-term features (legacy method)."""
        self.update_features_split(feat, img=img, allow_long=True)
    
    def _update_short_features(self, feat, img=None):
        """Always update short-term features."""
        try:
            eps = 1e-12
            f = feat.astype(np.float32, copy=False)
            f /= (np.linalg.norm(f) + eps)
            
            self.curr_feat = f
            beta = getattr(self, "update_ratio", 0.1)
            if self.smooth_feat is None:
                self.smooth_feat = f.copy()
            else:   
                self.smooth_feat = (1.0 - beta) * self.smooth_feat + beta * f
                self.smooth_feat /= (np.linalg.norm(self.smooth_feat) + eps)
                
            self.features.append(f)
        except Exception as e:
            print(f"error updating short features: {e}")
            
    def _update_long_features(self, feat, img=None):
        """Update long-term features when allowed by tracker."""
        try:
            self._ensure_long_buffers()
            eps = 1e-12
            f = feat.astype(np.float32, copy=False)
            f /= (np.linalg.norm(f) + eps)
            
            # Rotating buffer + cumulative centroid
            if len(self.long_feats) == self.long_feats.maxlen:
                oldest = self.long_feats.popleft()
                if self._long_sum is None:
                    self._long_sum = np.zeros_like(f, dtype=np.float32)
                self._long_sum -= oldest
                self._long_len = max(0, self._long_len - 1)
            
            self.long_feats.append(f)
            if self._long_sum is None:
                self._long_sum = f.copy()
                self._long_len = 1
            else:
                self._long_sum += f
                self._long_len += 1
                
            mean_f = self._long_sum / max(self._long_len, 1)
            self.long_feat_mean = mean_f / (np.linalg.norm(mean_f) + eps)
            
            # Update image only when long featuresare update
            if img is not None:
                try:
                    self.latest_image = crop_image(img, self.xyxy)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error updating long features: {e}")
    
    def update_features_split(self, feat, img=None, allow_long=True):
        if feat is None:
            return
        self._update_short_features(feat, img=img)
        if allow_long:
            # cho phép long theo stride để tránh spam
            if not hasattr(self, "frame_id") or (self.frame_id % getattr(self, "long_stride", 2) == 0):
                self._update_long_features(feat, img=img)
    
    def update_cls(self, cls, conf):
        """Update class history based on detection confidence."""
        try:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += conf
                    found = True
                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, conf])
                self.cls = cls
        except Exception as e:
            print(f"Error updating class: {e}")
            self.cls = cls

    def predict(self):
        """Predict the next state using Kalman filter."""
        try:
            if self.mean is None or self.kalman_filter is None:
                return
            mean_state = self.mean.copy()
            if self.state != TrackState.Tracked:
                mean_state[6:8] = 0  # Reset velocities
            self.mean, self.covariance = self.kalman_filter.predict(
                mean_state, self.covariance
            )
        except Exception as e:
            print(f"Error in track prediction: {e}")

    @staticmethod
    def multi_predict(stracks):
        """Perform batch prediction for multiple tracks."""
        if not stracks:
            return
        try:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
        
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6:8] = 0  # Reset velocities

            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for st, mean, cov in zip(stracks, multi_mean, multi_covariance):
                st.mean, st.covariance = mean, cov
        except Exception as e:
            print(f"Error in multi_predict: {e}")
            # Fallback to individual predictions
            for st in stracks:
                st.predict()

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Apply geometric motion compensation to multiple tracks."""
        if not stracks:
            return
        try:
            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4), R)
            t = H[:2, 2]

            for st in stracks:
                if st.mean is not None and st.covariance is not None:
                    mean = R8x8.dot(st.mean)
                    mean[:2] += t
                    st.mean = mean
                    st.covariance = R8x8.dot(st.covariance).dot(R8x8.T)
        except Exception as e:
            print(f"Error in multi_gmc: {e}")

    def activate(self, kalman_filter, frame_id, assign_id=True, img=None):
        """Activate the track."""
        try:
            self.kalman_filter = kalman_filter
            forced_id = getattr(self, "_forced_id", None)
            if forced_id is not None:
                self.id = int(forced_id)
                try:
                    delattr(self, "_forced_id")
                except Exception:
                    pass
            elif assign_id:
                if not hasattr(self, "id") or self.id is None or self.id < 0:
                    self.id = self.next_id()
            else:
                self.id = -1

            self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)
            self.tracklet_len = 0
            self.state        = TrackState.Tracked
            self.is_activated = True
            self.frame_id     = frame_id
            self.start_frame  = frame_id

            if self.curr_feat is not None and self.first_feat is None:
                self.first_feat = self.curr_feat.copy()
                
            if self.curr_feat is not None:
                old_stride = getattr(self, 'long_stride', 2)
                self.long_stride = 1
                self.update_features_split(self.curr_feat, img=img, allow_long=True)
                self.long_stride = old_stride

            self.just_initiated = True
        except Exception as e:
            print(f"Error activating track: {e}")

    def re_activate(self, new_track, frame_id, new_id=False, img=None, allow_feat=True, allow_long=True):
        """Re-activate a track with a new detection."""
        try:
            if self.mean is not None and self.covariance is not None and self.kalman_filter is not None:
                self.mean, self.covariance = self.kalman_filter.update(
                    self.mean, self.covariance, new_track.xywh
                )
            else:
                # Fallback initialization
                if self.kalman_filter is not None:
                    self.mean, self.covariance = self.kalman_filter.initiate(new_track.xywh)
           
            self.tracklet_len = 0
            self.state = TrackState.Tracked
            self.is_activated = True
            self.frame_id = frame_id
            if new_id:
                forced_id = getattr(self, "_forced_id", None)
                if forced_id is not None:
                    self.id = int(forced_id)
                    try:
                        delattr(self, "_forced_id")
                    except Exception:
                        pass
                else:
                    self.id = self.next_id()
                
            # Update features if allowed
            if allow_feat and getattr(new_track, "curr_feat", None) is not None:
                self.update_features_split(new_track.curr_feat, img=img, allow_long=allow_long)
                
            self.conf = getattr(new_track, 'conf', 1.0)
            self.cls = getattr(new_track, 'cls', 0)
            self.det_ind = getattr(new_track, 'det_ind', -1)
            self.update_cls(new_track.cls, new_track.conf)
            
            self.just_initiated = False
        except Exception as e:
            print(f"Error re-activating track: {e}")

    def update(self, new_track, frame_id, img=None, allow_feat=True, allow_long=True):
        """Update the current track with a matched detection."""
        try:
            self.frame_id = frame_id
            self.tracklet_len += 1
            self.history_observations.append(self.xyxy)

            if self.mean is not None and self.covariance is not None and self.kalman_filter is not None:
                self.mean, self.covariance = self.kalman_filter.update(
                    self.mean, self.covariance, new_track.xywh
                )

            if allow_feat and getattr(new_track, "curr_feat", None) is not None:
                self.update_features_split(new_track.curr_feat, img=img, allow_long=allow_long)
                
            self.state = TrackState.Tracked
            self.is_activated = True
            self.conf = getattr(new_track, 'conf', 1.0)
            self.cls = getattr(new_track, 'cls', 0)
            self.det_ind = getattr(new_track, 'det_ind', -1)
            self.update_cls(new_track.cls, new_track.conf)
            
            self.just_initiated = False
        except Exception as e:
            print(f"Error updating track: {e}")

    # Add into STrack
    def _get_kalman_state(self):
        return {
            "mean": None if self.mean is None else self.mean.copy(),
            "cov":  None if self.covariance is None else self.covariance.copy(),
        }

    def _set_kalman_state(self, state_dict):
        try:
            if state_dict is None: 
                return
            self.mean       = None if state_dict.get("mean") is None else state_dict["mean"].copy()
            self.covariance = None if state_dict.get("cov")  is None else state_dict["cov"].copy()
        except Exception as e:
            print(f"Error restoring KF state: {e}")

    @property
    def xyxy(self):
        """Convert bounding box format to (min x, min y, max x, max y)."""
        try:
            ret = self.mean[:4].copy() if self.mean is not None else self.xywh.copy()
            return xywh2xyxy(ret)
        except Exception:
            # Fallback to original xywh
            return xywh2xyxy(self.xywh.astype(np.float32)) if hasattr(self, 'xywh') else np.array([0,0,1,1], dtype=np.float32)

def crop_image(img, bbox, pad=2):
    """Crop image based on bounding box with padding."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = img.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        if x2 <= x1 or y2 <= y1:
            # Invalid crop region
            return np.zeros((1, 1, img.shape[2] if len(img.shape) > 2 else 1), dtype=img.dtype)
            
        return img[y1:y2, x1:x2].copy()
    except Exception as e:
        print(f"Error cropping image: {e}")
