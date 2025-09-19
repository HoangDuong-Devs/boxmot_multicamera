"""Global ID assignment utilities for multi-camera tracking.

The manager keeps a lightweight registry mapping each (camera_id, local_track_id)
pair to a global identity. A cosine-similarity based heuristic is used when
feature vectors are available so different cameras can agree on the same person.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class EntityRecord:
    """Stores information about an assigned global entity."""

    entity_id: int
    prototype: Optional[np.ndarray] = None
    last_feature: Optional[np.ndarray] = None
    last_timestamp: int = 0
    cameras: set = field(default_factory=set)
    observations: int = 0

    def update(self, feature: Optional[np.ndarray], timestamp: Optional[int], camera_id: str) -> None:
        self.cameras.add(str(camera_id))
        if feature is not None:
            v = np.asarray(feature, dtype=np.float32).ravel()
            n = np.linalg.norm(v)
            if np.isfinite(n) and n > 1e-6:
                v = v / n
                self.last_feature = v
                if self.prototype is None:
                    self.prototype = v.copy()
                else:
                    # Exponential moving average keeps prototype stable
                    self.prototype = 0.9 * self.prototype + 0.1 * v
                    pn = np.linalg.norm(self.prototype)
                    if np.isfinite(pn) and pn > 1e-6:
                        self.prototype = self.prototype / pn
        if timestamp is not None:
            self.last_timestamp = int(timestamp)
        self.observations += 1


class GlobalIDManager:
    """Assigns global IDs across cameras using appearance similarity."""

    def __init__(self, sim_threshold: float = 0.55):
        self.sim_threshold = float(sim_threshold)
        self._entities: Dict[int, EntityRecord] = {}
        self._lookup: Dict[Tuple[str, int], int] = {}
        self._pair_similarity: Dict[Tuple[str, int], float] = {}
        self._next_entity = 1

    def reset(self) -> None:
        self._entities.clear()
        self._lookup.clear()
        self._pair_similarity.clear()
        self._next_entity = 1

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        return self._lookup.get((str(camera_id), int(local_track_id)))

    def mark_lost(self, camera_id: str, local_track_id: int) -> None:
        key = (str(camera_id), int(local_track_id))
        entity_id = self._lookup.pop(key, None)
        if entity_id is None:
            return
        self._pair_similarity.pop(key, None)
        # keep entity record in case it reappears from another camera
        record = self._entities.get(entity_id)
        if record is not None:
            record.cameras.discard(str(camera_id))

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-6
        nb = np.linalg.norm(b) + 1e-6
        return float(np.dot(a, b) / (na * nb))

    def _match_entity(self, feature: Optional[np.ndarray]) -> Tuple[Optional[int], float]:
        if feature is None:
            return None, float("nan")
        v = np.asarray(feature, dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if not (np.isfinite(n) and n > 1e-6):
            return None, float("nan")
        v = v / n

        best_id, best_sim = None, -1.0
        for entity_id, rec in self._entities.items():
            if rec.prototype is None:
                continue
            sim = self._cosine(v, rec.prototype)
            if sim > best_sim:
                best_id, best_sim = entity_id, sim
        if best_id is not None and best_sim >= self.sim_threshold:
            return best_id, float(best_sim)
        return None, float(best_sim)

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        feature: Optional[np.ndarray] = None,
        timestamp: Optional[int] = None,
    ) -> int:
        key = (str(camera_id), int(local_track_id))
        existing = self._lookup.get(key)
        if existing is not None:
            record = self._entities.get(existing)
            sim = float("nan")
            if record is not None:
                record.update(feature, timestamp, camera_id)
                if feature is not None and record.prototype is not None:
                    sim = self._cosine(
                        np.asarray(feature, dtype=np.float32).ravel(), record.prototype
                    )
            if np.isnan(sim):
                sim = float("nan")
            self._pair_similarity[key] = sim
            return existing

        entity_id, match_sim = self._match_entity(feature)
        if entity_id is None:
            entity_id = self._next_entity
            self._next_entity += 1
            self._entities[entity_id] = EntityRecord(entity_id=entity_id)
            # treat new entity similarity as 1.0 if feature present else NaN
            match_sim = 1.0 if feature is not None else float("nan")

        self._lookup[key] = entity_id
        self._entities[entity_id].update(feature, timestamp, camera_id)
        self._pair_similarity[key] = float(match_sim)
        return entity_id

    def iter_entities(self):
        return self._entities.items()

    def get_similarity(self, camera_id: str, local_track_id: int) -> Optional[float]:
        key = (str(camera_id), int(local_track_id))
        return self._pair_similarity.get(key)
