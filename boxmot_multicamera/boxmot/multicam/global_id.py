
"""Global ID assignment utilities for multi-camera tracking.

The manager keeps a lightweight registry mapping each (camera_id, local_track_id)
pair to a global identity. A cosine-similarity based heuristic is used when
feature vectors are available so different cameras can agree on the same person.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from boxmot.utils.matching import linear_assignment


@dataclass
class EntityRecord:
    """Stores information about an assigned global entity."""

    entity_id: int
    prototype: Optional[np.ndarray] = None
    last_feature: Optional[np.ndarray] = None
    last_timestamp: int = 0
    last_position: Optional[np.ndarray] = None
    last_camera: Optional[str] = None
    cameras: set = field(default_factory=set)
    observations: int = 0

    def update(
        self,
        feature: Optional[np.ndarray],
        timestamp: Optional[int],
        camera_id: str,
        position: Optional[Sequence[float]] = None,
    ) -> None:
        self.cameras.add(str(camera_id))
        self.last_camera = str(camera_id)
        if feature is not None:
            v = np.asarray(feature, dtype=np.float32).ravel()
            n = np.linalg.norm(v)
            if np.isfinite(n) and n > 1e-6:
                v = v / n
                self.last_feature = v
                if self.prototype is None:
                    self.prototype = v.copy()
                else:
                    self.prototype = 0.9 * self.prototype + 0.1 * v
                    pn = np.linalg.norm(self.prototype)
                    if np.isfinite(pn) and pn > 1e-6:
                        self.prototype = self.prototype / pn
        if timestamp is not None:
            self.last_timestamp = int(timestamp)
        if position is not None:
            p = np.asarray(position, dtype=np.float32).ravel()
            if p.size >= 2 and np.isfinite(p).all():
                self.last_position = p[:2].copy()
        self.observations += 1


class GlobalIDManager:
    """Assign global IDs across cameras using appearance and optional spatial clues."""

    def __init__(
        self,
        sim_threshold: float = 0.55,
        sim_margin: float = 0.05,
        max_speed_px_per_frame: Optional[float] = None,
        max_reid_age: Optional[int] = None,
        reuse_delta: float = 0.1,
    ):
        self.sim_threshold = float(sim_threshold)
        self.sim_margin = float(sim_margin)
        self.max_speed_px_per_frame = (
            float(max_speed_px_per_frame) if max_speed_px_per_frame is not None else None
        )
        self.max_reid_age = int(max_reid_age) if max_reid_age is not None else None
        self.reuse_delta = float(max(reuse_delta, 0.0))

        self._entities: Dict[int, EntityRecord] = {}
        self._lookup: Dict[Tuple[str, int], int] = {}
        self._pair_similarity: Dict[Tuple[str, int], float] = {}
        self._next_entity = 1

        # Per-camera registry of active entity assignments (entity_id -> local_track_id)
        self._camera_entities: Dict[str, Dict[int, int]] = {}
        # Recently released assignments with short grace period
        self._recently_released: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._release_grace = 5

    def reset(self) -> None:
        self._entities.clear()
        self._lookup.clear()
        self._pair_similarity.clear()
        self._next_entity = 1
        self._camera_entities.clear()
        self._recently_released.clear()

    def _normalize_feature(self, feature: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if feature is None:
            return None
        v = np.asarray(feature, dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if not (np.isfinite(n) and n > 1e-6):
            return None
        return v / n

    def _sanitize_position(self, position: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if position is None:
            return None
        p = np.asarray(position, dtype=np.float32).ravel()
        if p.size < 2:
            return None
        p = p[:2]
        if not np.isfinite(p).all():
            return None
        return p

    def _score_candidate(
        self,
        record: EntityRecord,
        norm_feature: Optional[np.ndarray],
        position: Optional[np.ndarray],
        timestamp: Optional[int],
    ) -> float:
        if norm_feature is None or record.prototype is None:
            return -1.0

        sim = float(np.dot(norm_feature, record.prototype))

        if (
            sim >= self.sim_threshold
            and self.max_speed_px_per_frame is not None
            and position is not None
            and record.last_position is not None
            and timestamp is not None
        ):
            dt = max(1, int(timestamp) - int(record.last_timestamp))
            if self.max_reid_age is None or (int(timestamp) - int(record.last_timestamp) <= self.max_reid_age):
                dist = float(np.linalg.norm(position - record.last_position))
                speed = dist / float(dt)
                if speed > float(self.max_speed_px_per_frame):
                    return -1.0
        return sim

    def _entity_available(
        self,
        camera_id: str,
        entity_id: int,
        local_track_id: int,
        timestamp: Optional[int],
    ) -> bool:
        cam_map = self._camera_entities.get(camera_id)
        if cam_map is not None:
            owner = cam_map.get(int(entity_id))
            if owner is not None and owner != int(local_track_id):
                return False
        release = self._recently_released.get((camera_id, int(entity_id)))
        if release is None:
            return True
        _, released_ts = release
        if timestamp is None:
            return False
        if (timestamp - released_ts) < self._release_grace:
            return False
        self._recently_released.pop((camera_id, int(entity_id)), None)
        return True

    def _register_assignment(self, camera_id: str, local_track_id: int, entity_id: int) -> None:
        cam_map = self._camera_entities.setdefault(camera_id, {})
        cam_map[int(entity_id)] = int(local_track_id)
        self._recently_released.pop((camera_id, int(entity_id)), None)

    def _unregister_assignment(self, camera_id: str, entity_id: int, local_track_id: int) -> None:
        cam_map = self._camera_entities.get(camera_id)
        if cam_map is None:
            return
        owner = cam_map.get(int(entity_id))
        if owner == int(local_track_id):
            cam_map.pop(int(entity_id), None)
        if not cam_map:
            self._camera_entities.pop(camera_id, None)

    def _purge_releases(self, current_timestamp: Optional[int]) -> None:
        if current_timestamp is None:
            return
        purge = [
            key
            for key, (_, released_ts) in self._recently_released.items()
            if (current_timestamp - released_ts) >= self._release_grace
        ]
        for key in purge:
            self._recently_released.pop(key, None)

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        return self._lookup.get((str(camera_id), int(local_track_id)))

    def mark_lost(self, camera_id: str, local_track_id: int) -> None:
        key = (str(camera_id), int(local_track_id))
        entity_id = self._lookup.pop(key, None)
        if entity_id is None:
            return
        self._pair_similarity.pop(key, None)
        self._unregister_assignment(str(camera_id), int(entity_id), int(local_track_id))
        record = self._entities.get(entity_id)
        last_ts = int(record.last_timestamp) if record is not None else 0
        self._recently_released[(str(camera_id), int(entity_id))] = (int(local_track_id), last_ts)
        if record is not None:
            record.cameras.discard(str(camera_id))

    def _match_entity(
        self,
        feature: Optional[np.ndarray],
        position: Optional[Sequence[float]],
        timestamp: Optional[int],
        camera_id: str,
    ) -> Tuple[Optional[int], float]:
        norm_feature = self._normalize_feature(feature)
        if norm_feature is None:
            return None, float("nan")

        pos = self._sanitize_position(position)

        best_id, best_sim = None, -1.0
        sims = []
        for entity_id, rec in self._entities.items():
            if rec.prototype is None:
                continue
            sim = self._score_candidate(rec, norm_feature, pos, timestamp)
            sims.append((entity_id, sim))
            if sim > best_sim:
                best_id, best_sim = entity_id, sim

        if best_id is not None and best_sim >= self.sim_threshold:
            second_best_valid = max(
                (s for eid, s in sims if eid != best_id and s >= self.sim_threshold),
                default=-1.0,
            )
            if second_best_valid >= 0 and (best_sim - second_best_valid) < self.sim_margin:
                return None, float(best_sim)
            return best_id, float(best_sim)
        return None, float(best_sim)

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        feature: Optional[np.ndarray] = None,
        timestamp: Optional[int] = None,
        position: Optional[Sequence[float]] = None,
    ) -> int:
        camera_id_str = str(camera_id)
        key = (camera_id_str, int(local_track_id))
        self._purge_releases(timestamp)

        existing = self._lookup.get(key)
        if existing is not None:
            record = self._entities.get(existing)
            sim = float("nan")
            if record is not None:
                record.update(feature, timestamp, camera_id, position=position)
                if feature is not None and record.prototype is not None:
                    norm_feat = self._normalize_feature(feature)
                    if norm_feat is not None:
                        sim = float(np.dot(norm_feat, record.prototype))
            if np.isnan(sim):
                sim = float("nan")
            self._pair_similarity[key] = sim
            self._register_assignment(camera_id_str, int(local_track_id), int(existing))
            return existing

        entity_id, match_sim = self._match_entity(feature, position, timestamp, camera_id_str)
        if entity_id is not None and not self._entity_available(camera_id_str, int(entity_id), int(local_track_id), timestamp):
            entity_id = None
        if entity_id is None:
            entity_id = self._next_entity
            self._next_entity += 1
            self._entities[entity_id] = EntityRecord(entity_id=entity_id)
            match_sim = 1.0 if feature is not None else float("nan")

        self._lookup[key] = entity_id
        self._entities[entity_id].update(feature, timestamp, camera_id, position=position)
        self._pair_similarity[key] = float(match_sim)
        self._register_assignment(camera_id_str, int(local_track_id), int(entity_id))
        return entity_id

    def assign_batch(
        self,
        camera_id: str,
        local_track_ids: Sequence[int],
        features: Sequence[Optional[np.ndarray]],
        timestamp: Optional[int] = None,
        positions: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> Dict[int, int]:
        camera_id_str = str(camera_id)
        track_ids = [int(tid) for tid in local_track_ids]
        n_tracks = len(track_ids)
        if n_tracks == 0:
            return {}

        features_list = list(features)
        if len(features_list) != n_tracks:
            raise ValueError("features length must match track ids length")

        if positions is None:
            positions_list: list[Optional[Sequence[float]]] = [None] * n_tracks
        else:
            if len(positions) != n_tracks:
                raise ValueError("positions length must match track ids length")
            positions_list = list(positions)

        norm_features = [self._normalize_feature(f) for f in features_list]
        sanitized_positions = [self._sanitize_position(p) for p in positions_list]

        self._purge_releases(timestamp)

        prev_assignments = {
            tid: self._lookup.get((camera_id_str, tid)) for tid in track_ids
        }

        for tid in track_ids:
            key = (camera_id_str, tid)
            prev_entity = self._lookup.pop(key, None)
            if prev_entity is not None:
                self._unregister_assignment(camera_id_str, int(prev_entity), tid)
            self._pair_similarity.pop(key, None)

        candidate_items = list(self._entities.items())
        candidate_ids = [eid for eid, _ in candidate_items]
        sims_matrix = None
        if candidate_items:
            sims_matrix = np.full((n_tracks, len(candidate_items)), -1.0, dtype=np.float32)
            for row_idx, (norm_feat, pos) in enumerate(zip(norm_features, sanitized_positions)):
                if norm_feat is None:
                    continue
                for col_idx, (entity_id, record) in enumerate(candidate_items):
                    sims_matrix[row_idx, col_idx] = self._score_candidate(
                        record, norm_feat, pos, timestamp
                    )

        assigned: Dict[int, Tuple[int, float]] = {}
        if sims_matrix is not None and sims_matrix.size > 0:
            cost_matrix = np.full_like(sims_matrix, 1.0 + 1e-3)
            valid_mask = sims_matrix >= self.sim_threshold
            cost_matrix[valid_mask] = 1.0 - sims_matrix[valid_mask]

            matches, _, _ = linear_assignment(cost_matrix, thresh=1.0)
            if matches.size:
                for row_idx, col_idx in matches:
                    if not (0 <= row_idx < n_tracks and 0 <= col_idx < len(candidate_items)):
                        continue
                    sim_val = float(sims_matrix[row_idx, col_idx])
                    if sim_val < self.sim_threshold:
                        continue
                    entity_id = candidate_ids[col_idx]
                    tid = track_ids[row_idx]
                    if not self._entity_available(camera_id_str, int(entity_id), tid, timestamp):
                        continue
                    if self.sim_margin > 0:
                        other_sims = [
                            float(sims_matrix[row_idx, j])
                            for j in range(len(candidate_items))
                            if j != col_idx and sims_matrix[row_idx, j] >= self.sim_threshold
                        ]
                        if other_sims:
                            second_best = max(other_sims)
                            if (sim_val - second_best) < self.sim_margin:
                                continue
                    assigned[row_idx] = (col_idx, sim_val)

        results: Dict[int, int] = {}
        used_entities: set[int] = set()

        for row_idx, tid in enumerate(track_ids):
            key = (camera_id_str, tid)
            feature = features_list[row_idx]
            norm_feat = norm_features[row_idx]
            position_raw = positions_list[row_idx]
            pos_clean = sanitized_positions[row_idx]

            if row_idx in assigned:
                col_idx, sim_val = assigned[row_idx]
                entity_id = candidate_ids[col_idx]
                if (
                    entity_id in used_entities
                    or not self._entity_available(camera_id_str, int(entity_id), tid, timestamp)
                ):
                    assigned.pop(row_idx, None)
                else:
                    record = self._entities.get(entity_id)
                    if record is not None:
                        record.update(feature, timestamp, camera_id_str, position=position_raw)
                        self._lookup[key] = entity_id
                        self._pair_similarity[key] = sim_val
                        used_entities.add(entity_id)
                        self._register_assignment(camera_id_str, tid, int(entity_id))
                        results[tid] = entity_id
                        continue

            prev_gid = prev_assignments.get(tid)
            if (
                prev_gid is not None
                and prev_gid in self._entities
                and prev_gid not in used_entities
                and self._entity_available(camera_id_str, int(prev_gid), tid, timestamp)
            ):
                record = self._entities[prev_gid]
                reuse_ok = False
                sim_for_log = float("nan")
                if norm_feat is None:
                    reuse_ok = True
                else:
                    reuse_threshold = max(self.sim_threshold - self.reuse_delta, -1.0)
                    prev_sim = self._score_candidate(record, norm_feat, pos_clean, timestamp)
                    if prev_sim >= reuse_threshold:
                        reuse_ok = True
                        sim_for_log = prev_sim
                if reuse_ok:
                    record.update(feature, timestamp, camera_id_str, position=position_raw)
                    self._lookup[key] = prev_gid
                    if norm_feat is not None and record.prototype is not None:
                        sim_for_log = float(np.dot(norm_feat, record.prototype))
                    self._pair_similarity[key] = sim_for_log
                    used_entities.add(prev_gid)
                    self._register_assignment(camera_id_str, tid, int(prev_gid))
                    results[tid] = prev_gid
                    continue

            new_entity_id = self._next_entity
            self._next_entity += 1
            self._entities[new_entity_id] = EntityRecord(entity_id=new_entity_id)
            self._lookup[key] = new_entity_id
            self._entities[new_entity_id].update(feature, timestamp, camera_id_str, position=position_raw)
            sim_val = 1.0 if norm_feat is not None else float("nan")
            self._pair_similarity[key] = sim_val
            used_entities.add(new_entity_id)
            self._register_assignment(camera_id_str, tid, int(new_entity_id))
            results[tid] = new_entity_id

        return results

    def iter_entities(self):
        return self._entities.items()

    def get_similarity(self, camera_id: str, local_track_id: int) -> Optional[float]:
        key = (str(camera_id), int(local_track_id))
        return self._pair_similarity.get(key)
