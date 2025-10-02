
"""Enhanced Global ID assignment utilities for multi-camera tracking.

The manager maintains global identities across cameras with:
- Per-camera state tracking (active/lost/inactive)
- Direct matching with global entities instead of grace periods
- Robust conflict resolution and coexistence tracking
- Persistent historical evidence via Qdrant integration

Each global ID represents a persistent identity across all cameras,
with state tracked separately per camera to handle overlapping views.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Literal, Set, List
from enum import Enum
import time

import numpy as np
from scipy.spatial.distance import cdist

from boxmot.utils.matching import linear_assignment

def compute_cosine_cost(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Compute cosine cost between two features (like BotSORT).
    
    Args:
        feat1: First feature vector
        feat2: Second feature vector
        
    Returns:
        Cost value (0.0 = identical, 1.0 = completely different)
    """
    if feat1 is None or feat2 is None:
        return 1.0
    
    # Ensure features are normalized
    feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
    feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(feat1_norm, feat2_norm)
    
    # Convert to cost: cost = 0.5 * (1.0 - similarity) like BotSORT
    cost = 0.5 * (1.0 - similarity)
    return max(0.0, min(1.0, cost))  # Clamp to [0, 1]

# Optional Qdrant integration
try:
    from boxmot.qdrant.qdrant_long_reid import HybridLongBank
    _HAVE_QDRANT = True
except ImportError:
    _HAVE_QDRANT = False
    HybridLongBank = None


class CameraStatus(Enum):
    """Status of a global entity in a specific camera."""
    ACTIVE = "active"      # Currently being tracked
    LOST = "lost"          # Recently lost, may reappear
    INACTIVE = "inactive"  # Not seen for a long time
    NEVER_SEEN = "never_seen"  # Never appeared in this camera


@dataclass
class CameraState:
    """Per-camera state for a global entity."""
    status: CameraStatus = CameraStatus.NEVER_SEEN
    local_id: Optional[int] = None
    last_timestamp: int = 0
    last_position: Optional[np.ndarray] = None
    confidence: float = 0.0
    observations: int = 0
    
    def update(self, local_id: Optional[int], timestamp: int, position: Optional[np.ndarray] = None):
        """Update camera state with new information."""
        self.local_id = local_id
        self.last_timestamp = timestamp
        if position is not None:
            self.last_position = np.asarray(position, dtype=np.float32)[:2].copy()
        self.observations += 1
        
    def mark_lost(self, timestamp: int):
        """Mark as lost at given timestamp."""
        self.status = CameraStatus.LOST
        self.local_id = None
        self.last_timestamp = timestamp
        
    def mark_inactive(self, timestamp: int):
        """Mark as inactive (long-term lost)."""
        self.status = CameraStatus.INACTIVE
        self.local_id = None
        self.last_timestamp = timestamp


@dataclass
class GlobalEntity:
    """Enhanced global entity with per-camera state tracking."""
    
    global_id: int
    prototype: Optional[np.ndarray] = None
    last_feature: Optional[np.ndarray] = None
    total_observations: int = 0
    created_timestamp: int = field(default_factory=lambda: int(time.time()))
    last_updated: int = field(default_factory=lambda: int(time.time()))
    
    # Per-camera states
    camera_states: Dict[str, CameraState] = field(default_factory=dict)
    
    # Coexistence tracking
    coexistence_history: Dict[int, int] = field(default_factory=dict)  # global_id -> last_timestamp
    
    # Feature aggregation from multiple cameras
    feature_history: List[Tuple[np.ndarray, str, int]] = field(default_factory=list)  # (feature, camera, timestamp)
    max_history_size: int = 50
    
    def get_camera_state(self, camera_id: str) -> CameraState:
        """Get or create camera state for given camera."""
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = CameraState()
        return self.camera_states[camera_id]
    
    def update_feature(self, feature: Optional[np.ndarray], camera_id: str, timestamp: int):
        """Update prototype with new feature from specific camera."""
        if feature is None:
            return
            
        v = np.asarray(feature, dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if not (np.isfinite(n) and n > 1e-6):
            return
        v = v / n
        
        self.last_feature = v
        self.last_updated = timestamp
        
        # Add to feature history
        self.feature_history.append((v, camera_id, timestamp))
        if len(self.feature_history) > self.max_history_size:
            self.feature_history.pop(0)
        
        # Update prototype with weighted average
        if self.prototype is None:
            self.prototype = v.copy()
        else:
            # Weight recent features more heavily
            alpha = 0.2  # Learning rate
            self.prototype = (1 - alpha) * self.prototype + alpha * v
            pn = np.linalg.norm(self.prototype)
            if np.isfinite(pn) and pn > 1e-6:
                self.prototype = self.prototype / pn
    
    def get_cost_to_feature(self, feature: np.ndarray) -> float:
        """Calculate cost between prototype and given feature (like BotSORT)."""
        if self.prototype is None:
            return 1.0
        
        v = np.asarray(feature, dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if not (np.isfinite(n) and n > 1e-6):
            return 1.0
        v = v / n
        
        similarity = float(np.dot(self.prototype, v))
        # Convert to cost: cost = 0.5 * (1.0 - similarity) like BotSORT
        cost = 0.5 * (1.0 - similarity)
        return max(0.0, min(1.0, cost))

    def get_similarity_to_feature(self, feature: np.ndarray) -> float:
        """Calculate similarity between prototype and given feature."""
        if self.prototype is None:
            return -1.0
        
        v = np.asarray(feature, dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if not (np.isfinite(n) and n > 1e-6):
            return -1.0
        v = v / n
        
        return float(np.dot(self.prototype, v))
    
    def is_active_in_camera(self, camera_id: str) -> bool:
        """Check if entity is currently active in given camera."""
        state = self.camera_states.get(camera_id)
        return state is not None and state.status == CameraStatus.ACTIVE
    
    def is_lost_in_camera(self, camera_id: str) -> bool:
        """Check if entity is lost in given camera."""
        state = self.camera_states.get(camera_id)
        return state is not None and state.status == CameraStatus.LOST
    
    def get_active_cameras(self) -> Set[str]:
        """Get set of cameras where this entity is currently active."""
        return {
            camera_id for camera_id, state in self.camera_states.items()
            if state.status == CameraStatus.ACTIVE
        }
    
    def update_coexistence(self, other_global_id: int, timestamp: int):
        """Record coexistence with another global entity."""
        self.coexistence_history[other_global_id] = timestamp
    
    def check_coexistence_conflict(self, other_global_ids: Set[int], current_time: int, ttl_seconds: int = 3600) -> bool:
        """Check if assigning this entity would conflict with currently active entities."""
        for other_id in other_global_ids:
            if other_id in self.coexistence_history:
                last_coex_time = self.coexistence_history[other_id]
                if (current_time - last_coex_time) < ttl_seconds:
                    return True
        return False


class EnhancedGlobalIDManager:
    """Enhanced Global ID manager with per-camera state tracking and direct entity matching.
    
    Key improvements:
    - Per-camera state tracking (active/lost/inactive)
    - Direct matching with global entities instead of grace periods
    - Robust coexistence conflict detection
    - Integration with Qdrant for persistent historical evidence
    """

    def __init__(
        self,
        # Backward compatibility - accept both sim_threshold and cost_threshold
        sim_threshold: Optional[float] = None,
        cost_threshold: Optional[float] = None,
        sim_margin: float = 0.05,
        cost_margin: float = 0.05,
        max_speed_px_per_frame: Optional[float] = None,
        max_reid_age: Optional[int] = None,
        # Enhanced parameters
        lost_timeout_seconds: int = 300,  # 5 minutes
        inactive_timeout_seconds: int = 3600,  # 1 hour
        coexistence_ttl_seconds: int = 1800,  # 30 minutes
        # Qdrant integration
        use_qdrant: bool = True,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "enhanced_global_reid",
        reid_dim: int = 512,
        historical_sim_threshold: Optional[float] = None,
        historical_cost_threshold: Optional[float] = None,
    ):
        # Handle backward compatibility for thresholds
        if cost_threshold is not None:
            self.cost_threshold = float(cost_threshold)
        elif sim_threshold is not None:
            # Convert similarity to cost: cost = 1.0 - similarity  
            self.cost_threshold = 1.0 - float(sim_threshold)
        else:
            self.cost_threshold = 0.4  # Default cost threshold
            
        # Handle margins
        if cost_threshold is not None:
            self.cost_margin = float(cost_margin)
        else:
            self.cost_margin = float(sim_margin)
            
        # Historical thresholds
        if historical_cost_threshold is not None:
            self.historical_cost_threshold = float(historical_cost_threshold)
        elif historical_sim_threshold is not None:
            self.historical_cost_threshold = 1.0 - float(historical_sim_threshold)
        else:
            self.historical_cost_threshold = 0.3  # Default historical cost threshold
        self.max_speed_px_per_frame = (
            float(max_speed_px_per_frame) if max_speed_px_per_frame is not None else None
        )
        self.max_reid_age = int(max_reid_age) if max_reid_age is not None else None
        
        # Enhanced parameters
        self.lost_timeout_seconds = lost_timeout_seconds
        self.inactive_timeout_seconds = inactive_timeout_seconds
        self.coexistence_ttl_seconds = coexistence_ttl_seconds
        self.historical_cost_threshold = historical_cost_threshold

        # Core data structures
        self._global_entities: Dict[int, GlobalEntity] = {}
        self._next_global_id = 1
        
        # Quick lookup: (camera_id, local_id) -> global_id
        self._local_to_global: Dict[Tuple[str, int], int] = {}
        
        # Track assignment confidence for logging
        self._assignment_logs: Dict[Tuple[str, int], Dict] = {}

        # Qdrant integration
        self.qdrant_bank = None
        if use_qdrant and _HAVE_QDRANT:
            try:
                self.qdrant_bank = HybridLongBank(
                    use_qdrant=True,
                    host=qdrant_host,
                    port=qdrant_port,
                    collection=qdrant_collection,
                    dim=reid_dim,
                    slots=100,
                )
                print(f"[EnhancedGlobalIDManager] Qdrant integration enabled: {self.qdrant_bank.backend_name}")
            except Exception as e:
                print(f"[EnhancedGlobalIDManager] Qdrant init failed, disabled: {e}")
                self.qdrant_bank = None
        else:
            print("[EnhancedGlobalIDManager] Qdrant integration disabled")
            self.qdrant_bank = None

    def reset(self) -> None:
        """Reset all state."""
        self._global_entities.clear()
        self._local_to_global.clear()
        self._assignment_logs.clear()
        self._next_global_id = 1

    def _normalize_feature(self, feature: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Normalize feature vector."""
        if feature is None:
            return None
        v = np.asarray(feature, dtype=np.float32).ravel()
        n = np.linalg.norm(v)
        if not (np.isfinite(n) and n > 1e-6):
            return None
        return v / n

    def _sanitize_position(self, position: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        """Sanitize position coordinates."""
        if position is None:
            return None
        p = np.asarray(position, dtype=np.float32).ravel()
        if p.size < 2:
            return None
        p = p[:2]
        if not np.isfinite(p).all():
            return None
        return p

    def _add_to_qdrant_history(self, global_id: int, feature: np.ndarray, camera_id: str, timestamp: int):
        """Add feature to Qdrant historical storage."""
        if self.qdrant_bank is None or feature is None:
            return
            
        try:
            v = self._normalize_feature(feature)
            if v is None:
                return
                
            extra = {
                "global_id": int(global_id),
                "camera_id": str(camera_id),
                "timestamp": int(timestamp),
                "frame_id": int(timestamp),
            }
            
            # Use timestamp-based slot distribution
            slot = (timestamp // 10) % 100
            
            self.qdrant_bank.add_ring(
                run_uid="enhanced_global",
                track_id=int(global_id),
                vector=v,
                slot=slot,
                extra=extra,
                camera_id=str(camera_id)
            )
        except Exception as e:
            print(f"[EnhancedGlobalIDManager] Failed to add to Qdrant: {e}")

    def _delete_qdrant_vectors(self, global_id: int, camera_id: str, old_local_id: int):
        """Delete old Qdrant vectors when local_id changes.
        
        When a track ID changes (e.g., track 1 → track 8 for same person),
        we need to remove the old vectors to avoid confusion.
        """
        if self.qdrant_bank is None:
            return
        
        try:
            # Delete all vectors for this (global_id, camera_id, old_local_id) combination
            # Note: Qdrant stores by (run_uid, track_id, camera_id)
            # We use global_id as track_id, so we need to delete vectors with matching metadata
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Filter: global_id AND camera_id (to delete all old vectors for this camera)
            delete_filter = Filter(
                must=[
                    FieldCondition(key="global_id", match=MatchValue(value=int(global_id))),
                    FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id))),
                ]
            )
            
            result = self.qdrant_bank.backend.client.delete(
                collection_name=self.qdrant_bank.backend.collection,
                points_selector=delete_filter
            )
            
            print(f"[GlobalIDManager] 🗑️  Deleted {result.operation_id if hasattr(result, 'operation_id') else 'old'} Qdrant vectors for Global {global_id}, camera {camera_id}")
            
        except Exception as e:
            print(f"[GlobalIDManager] ⚠️  Failed to delete old Qdrant vectors: {e}")

    def _search_qdrant_history(self, feature: np.ndarray, camera_id: Optional[str] = None) -> List[Tuple[int, float]]:
        """Search Qdrant for historical matches, return list of (global_id, similarity)."""
        if self.qdrant_bank is None or feature is None:
            return []
            
        try:
            v = self._normalize_feature(feature)
            if v is None:
                return []
                
            # Search with Qdrant backend directly
            if hasattr(self.qdrant_bank, 'backend') and hasattr(self.qdrant_bank.backend, 'client'):
                try:
                    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                    
                    must_conditions = []
                    if camera_id is not None:
                        must_conditions.append(
                            FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id)))
                        )
                    
                    search_filter = Filter(must=must_conditions) if must_conditions else None
                    
                    # Convert cost threshold to similarity threshold for Qdrant
                    # cost = 0.5 * (1.0 - similarity)
                    # => similarity = 1.0 - (cost / 0.5) = 1.0 - 2.0 * cost
                    if self.historical_cost_threshold is None:
                        # No historical threshold set, skip search
                        return []
                    
                    historical_sim_threshold = 1.0 - 2.0 * self.historical_cost_threshold
                    
                    results = self.qdrant_bank.backend.client.search(
                        collection_name=self.qdrant_bank.backend.collection,
                        query_vector=v.tolist(),
                        query_filter=search_filter,
                        limit=20,
                        with_payload=True,
                        score_threshold=historical_sim_threshold
                    )
                    
                    # Group by global_id and return best scores
                    global_id_scores = {}
                    for hit in results:
                        payload = hit.payload
                        gid = int(payload.get("global_id", -1))
                        if gid <= 0:
                            continue
                        score = float(hit.score)
                        if gid not in global_id_scores:
                            global_id_scores[gid] = []
                        global_id_scores[gid].append(score)
                    
                    # Return average scores for each global_id
                    return [
                        (gid, sum(scores) / len(scores))
                        for gid, scores in global_id_scores.items()
                    ]
                    
                except Exception as e:
                    print(f"[EnhancedGlobalIDManager] Qdrant search failed: {e}")
                    
        except Exception as e:
            print(f"[EnhancedGlobalIDManager] Historical search error: {e}")
        
        return []

    def _update_entity_states(self, current_time: Optional[int] = None):
        """Update entity states based on timeouts."""
        if current_time is None:
            current_time = int(time.time())
            
        for entity in self._global_entities.values():
            for camera_id, state in entity.camera_states.items():
                time_since_last = current_time - state.last_timestamp
                
                if state.status == CameraStatus.ACTIVE:
                    # Active entities should be updated externally, no timeout here
                    continue
                elif state.status == CameraStatus.LOST:
                    if time_since_last > self.inactive_timeout_seconds:
                        state.mark_inactive(current_time)
                elif state.status == CameraStatus.NEVER_SEEN:
                    # No change needed
                    continue

    def _get_active_global_ids(self, exclude_camera: Optional[str] = None) -> Set[int]:
        """Get all currently active global IDs, optionally excluding a specific camera."""
        active_ids = set()
        for global_id, entity in self._global_entities.items():
            for camera_id, state in entity.camera_states.items():
                if exclude_camera and camera_id == exclude_camera:
                    continue
                if state.status == CameraStatus.ACTIVE:
                    active_ids.add(global_id)
                    break  # Entity is active if active in any camera
        return active_ids

    def _get_matching_candidates(
        self, 
        camera_id: str, 
        feature: np.ndarray, 
        prioritize_lost: bool = True
    ) -> List[Tuple[int, float]]:
        """Get candidates for matching, prioritizing lost/inactive entities in same camera."""
        candidates = []
        
        # Priority 1: Lost entities in same camera
        lost_candidates = []
        # Priority 2: Inactive entities in same camera  
        inactive_candidates = []
        # Priority 3: Entities never seen in this camera
        never_seen_candidates = []
        # Priority 4: Active entities in other cameras (lowest priority)
        other_candidates = []
        
        for global_id, entity in self._global_entities.items():
            cost = entity.get_cost_to_feature(feature)
            
            # Debug log for each entity
            camera_state = entity.get_camera_state(camera_id)
            status_str = camera_state.status.value if camera_state else "none"
            print(f"[CANDIDATES_DEBUG]   Entity {global_id}: cost={cost:.4f}, status={status_str}, threshold={self.cost_threshold}")
            
            if cost > self.cost_threshold:
                print(f"[CANDIDATES_DEBUG]     -> SKIP: cost {cost:.4f} > threshold {self.cost_threshold}")
                continue
                
            camera_state = entity.get_camera_state(camera_id)
            
            if camera_state.status == CameraStatus.LOST:
                lost_candidates.append((global_id, cost))
                print(f"[CANDIDATES_DEBUG]     -> Added to LOST candidates (priority 1)")
            elif camera_state.status == CameraStatus.INACTIVE:
                inactive_candidates.append((global_id, cost))
                print(f"[CANDIDATES_DEBUG]     -> Added to INACTIVE candidates (priority 2)")
            elif camera_state.status == CameraStatus.NEVER_SEEN:
                never_seen_candidates.append((global_id, cost))
                print(f"[CANDIDATES_DEBUG]     -> Added to NEVER_SEEN candidates (priority 3)")
            else:  # ACTIVE - should not happen for new assignments
                other_candidates.append((global_id, cost))
                print(f"[CANDIDATES_DEBUG]     -> Added to OTHER candidates (priority 4)")
        
        # Sort each group by cost (ascending - lower cost is better)
        lost_candidates.sort(key=lambda x: x[1])
        inactive_candidates.sort(key=lambda x: x[1])
        never_seen_candidates.sort(key=lambda x: x[1])
        other_candidates.sort(key=lambda x: x[1])
        
        # Debug log candidate counts
        print(f"[CANDIDATES_DEBUG] Candidate summary:")
        print(f"[CANDIDATES_DEBUG]   LOST: {len(lost_candidates)} - {lost_candidates[:3]}")
        print(f"[CANDIDATES_DEBUG]   INACTIVE: {len(inactive_candidates)} - {inactive_candidates[:3]}")
        print(f"[CANDIDATES_DEBUG]   NEVER_SEEN: {len(never_seen_candidates)} - {never_seen_candidates[:3]}")
        print(f"[CANDIDATES_DEBUG]   OTHER: {len(other_candidates)} - {other_candidates[:3]}")
        
        # Combine in priority order
        if prioritize_lost:
            candidates.extend(lost_candidates)
            candidates.extend(inactive_candidates)
            candidates.extend(never_seen_candidates)
            candidates.extend(other_candidates)
        else:
            # For historical search, don't prioritize by camera state
            all_candidates = lost_candidates + inactive_candidates + never_seen_candidates + other_candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = all_candidates
            
        return candidates

    def _check_temporal_consistency(
        self, 
        global_id: int, 
        camera_id: str, 
        timestamp: int,
        position: Optional[np.ndarray] = None
    ) -> bool:
        """Check if assignment is temporally consistent."""
        entity = self._global_entities.get(global_id)
        if entity is None:
            return True
            
        camera_state = entity.get_camera_state(camera_id)
        
        # Check speed constraint if both position and previous position available
        if (self.max_speed_px_per_frame is not None and 
            position is not None and 
            camera_state.last_position is not None and
            camera_state.last_timestamp > 0):
            
            dt = max(1, timestamp - camera_state.last_timestamp)
            if self.max_reid_age is None or dt <= self.max_reid_age:
                dist = float(np.linalg.norm(position - camera_state.last_position))
                speed = dist / float(dt)
                if speed > float(self.max_speed_px_per_frame):
                    return False
                    
        return True

    def _validate_assignment(
        self, 
        global_id: int, 
        camera_id: str, 
        local_id: int,
        feature: Optional[np.ndarray],
        timestamp: int,
        position: Optional[np.ndarray] = None
    ) -> bool:
        """Comprehensive validation before assignment."""
        print(f"  [VALIDATION] Checking Global ID {global_id} for {camera_id}:{local_id}")
        
        entity = self._global_entities.get(global_id)
        if entity is None:
            print(f"  [VALIDATION] ❌ Entity {global_id} not found")
            return False
            
        # Check if already assigned to different local_id in same camera
        # Only reject if ACTIVE - allow reuse if LOST or INACTIVE
        camera_state = entity.get_camera_state(camera_id)
        print(f"  [VALIDATION] 🔍 Entity status={camera_state.status.value}, local_id={camera_state.local_id}")
        
        if (camera_state.status == CameraStatus.ACTIVE and 
            camera_state.local_id is not None and 
            camera_state.local_id != local_id):
            print(f"  [VALIDATION] ❌ Already assigned to different ACTIVE local_id: {camera_state.local_id} != {local_id}")
            return False
        
        # If entity is LOST/INACTIVE with different local_id, allow reuse (this is the reactivation case)
        if camera_state.local_id is not None and camera_state.local_id != local_id:
            if camera_state.status in [CameraStatus.LOST, CameraStatus.INACTIVE]:
                print(f"  [VALIDATION] ℹ️  Entity was {camera_state.status.value} with local_id={camera_state.local_id}, allowing reuse for new local_id={local_id}")
            
        # Check temporal consistency
        if not self._check_temporal_consistency(global_id, camera_id, timestamp, position):
            print(f"  [VALIDATION] ❌ Temporal consistency check failed")
            return False
            
        # Check coexistence conflicts
        active_global_ids = self._get_active_global_ids(exclude_camera=camera_id)
        if entity.check_coexistence_conflict(active_global_ids, timestamp, self.coexistence_ttl_seconds):
            print(f"  [VALIDATION] ❌ Coexistence conflict detected")
            return False
        
        print(f"  [VALIDATION] ✅ All checks passed for Global ID {global_id}")
        return True

    def match_new_local_track(
        self,
        camera_id: str,
        local_id: int,
        feature: Optional[np.ndarray],
        timestamp: int,
        position: Optional[Sequence[float]] = None
    ) -> Tuple[Optional[int], float, str]:
        """Match new local track to existing global entities.
        
        Returns:
            (global_id, similarity, match_type) where match_type is one of:
            'lost_reactivation', 'inactive_reactivation', 'cross_camera_match', 
            'historical_match', 'new_entity'
        """
        if feature is None:
            # Create new entity without feature matching
            return None, float("nan"), "new_entity"
            
        norm_feature = self._normalize_feature(feature)
        if norm_feature is None:
            return None, float("nan"), "new_entity"
            
        pos = self._sanitize_position(position)
        
        # Update entity states first
        self._update_entity_states(timestamp)
        
        # Get candidates with prioritization
        candidates = self._get_matching_candidates(camera_id, norm_feature, prioritize_lost=True)
        
        # Try to find valid assignment
        for global_id, cost in candidates:
            print(f"\n[MATCH_DEBUG] Trying candidate: Global ID {global_id}, cost={cost:.4f}")
            
            if self._validate_assignment(global_id, camera_id, local_id, norm_feature, timestamp, pos):
                entity = self._global_entities[global_id]
                camera_state = entity.get_camera_state(camera_id)
                
                # Determine match type
                if camera_state.status == CameraStatus.LOST:
                    match_type = "lost_reactivation"
                elif camera_state.status == CameraStatus.INACTIVE:
                    match_type = "inactive_reactivation"
                elif camera_state.status == CameraStatus.NEVER_SEEN:
                    # Check if active in other cameras
                    if entity.get_active_cameras():
                        match_type = "cross_camera_match"
                    else:
                        match_type = "inactive_reactivation"
                else:
                    match_type = "cross_camera_match"
                
                # Convert cost back to similarity for return (backward compatibility)
                similarity = 1.0 - cost
                print(f"[MATCH_DEBUG] ✅ VALIDATION PASSED: Global ID {global_id}")
                print(f"[MATCH_DEBUG]    Match type: {match_type}")
                print(f"[MATCH_DEBUG]    Cost: {cost:.4f}, Similarity: {similarity:.4f}")
                return global_id, similarity, match_type
            else:
                print(f"[MATCH_DEBUG] ❌ VALIDATION FAILED for Global ID {global_id}")        # No current entity matches, try historical search
        print(f"[MATCH_DEBUG] No current candidates matched, trying Qdrant historical search...")
        historical_matches = self._search_qdrant_history(norm_feature, camera_id)
        print(f"[MATCH_DEBUG] Historical matches found: {len(historical_matches)}")
        
        for global_id, similarity in historical_matches:
            # Convert similarity to cost for comparison
            cost = compute_cosine_cost(norm_feature, None)  # Will be computed properly in validation
            if cost <= self.historical_cost_threshold:
                # Check if this global_id still exists and can be reactivated
                if global_id in self._global_entities:
                    if self._validate_assignment(global_id, camera_id, local_id, norm_feature, timestamp, pos):
                        return global_id, similarity, "historical_match"
                else:
                    # Historical entity no longer exists, create new with same ID
                    if global_id >= self._next_global_id:
                        self._next_global_id = global_id + 1
                    return global_id, similarity, "historical_match"
        
        # No matches found, create new entity
        return None, float("nan"), "new_entity"

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        feature: Optional[np.ndarray] = None,
        timestamp: Optional[int] = None,
        position: Optional[Sequence[float]] = None,
        existing_track_ids: Optional[set[int]] = None,  # NEW: track IDs that exist in tracker
    ) -> int:
        """Assign global ID to local track with enhanced matching logic.
        
        Args:
            camera_id: Camera identifier
            local_track_id: Local track ID from single-camera tracker
            feature: Appearance feature vector
            timestamp: Current timestamp
            position: Track position [x, y]
            existing_track_ids: Set of track IDs that currently exist in tracker
                               (active_tracks + lost_tracks). Used to detect removed tracks.
        """
        camera_id_str = str(camera_id)
        local_id = int(local_track_id)
        timestamp = timestamp if timestamp is not None else int(time.time())
        
        # Sync: Mark entities as LOST if their local_id no longer exists in tracker
        if existing_track_ids is not None:
            for entity in self._global_entities.values():
                camera_state = entity.get_camera_state(camera_id_str)
                if (camera_state.status == CameraStatus.ACTIVE and 
                    camera_state.local_id is not None and
                    camera_state.local_id not in existing_track_ids):
                    print(f"[GlobalIDManager] Track {camera_id_str}:{camera_state.local_id} removed from tracker, marking Global ID {entity.global_id} as LOST")
                    camera_state.mark_lost(timestamp)
        
        # Check if already assigned
        existing_global_id = self._local_to_global.get((camera_id_str, local_id))
        if existing_global_id is not None:
            # Update existing assignment
            entity = self._global_entities.get(existing_global_id)
            if entity is not None:
                entity.update_feature(feature, camera_id_str, timestamp)
                camera_state = entity.get_camera_state(camera_id_str)
                camera_state.update(local_id, timestamp, self._sanitize_position(position))
                camera_state.status = CameraStatus.ACTIVE
                
                # Add to Qdrant history
                if feature is not None:
                    self._add_to_qdrant_history(existing_global_id, feature, camera_id_str, timestamp)
                
                # Log assignment info
                similarity = entity.get_similarity_to_feature(feature) if feature is not None else float("nan")
                self._assignment_logs[(camera_id_str, local_id)] = {
                    "global_id": existing_global_id,
                    "similarity": similarity,
                    "match_type": "existing_assignment",
                    "timestamp": timestamp
                }
                
                return existing_global_id
        
        # Try to match with existing global entities
        global_id, similarity, match_type = self.match_new_local_track(
            camera_id_str, local_id, feature, timestamp, position
        )
        
        if global_id is None:
            # Create new global entity
            global_id = self._next_global_id
            self._next_global_id += 1
            entity = GlobalEntity(global_id=global_id)
            self._global_entities[global_id] = entity
            match_type = "new_entity"
            similarity = 1.0 if feature is not None else float("nan")
        else:
            # Use existing or reactivated entity
            entity = self._global_entities.get(global_id)
            if entity is None:
                # Historical match for non-existent entity, recreate
                entity = GlobalEntity(global_id=global_id)
                self._global_entities[global_id] = entity
        
        # Update entity and camera state
        entity.update_feature(feature, camera_id_str, timestamp)
        camera_state = entity.get_camera_state(camera_id_str)
        
        # If local_id is changing, delete old Qdrant vectors
        old_local_id = camera_state.local_id
        if old_local_id is not None and old_local_id != local_id:
            print(f"[GlobalIDManager] 🔄 Local ID change detected: {camera_id_str}:{old_local_id} → {camera_id_str}:{local_id} (Global {global_id})")
            self._delete_qdrant_vectors(global_id, camera_id_str, old_local_id)
        
        camera_state.update(local_id, timestamp, self._sanitize_position(position))
        camera_state.status = CameraStatus.ACTIVE
        
        # Register assignment
        self._local_to_global[(camera_id_str, local_id)] = global_id
        
        # Add to Qdrant history
        if feature is not None:
            self._add_to_qdrant_history(global_id, feature, camera_id_str, timestamp)
        
        # Log assignment info
        self._assignment_logs[(camera_id_str, local_id)] = {
            "global_id": global_id,
            "similarity": similarity,
            "match_type": match_type,
            "timestamp": timestamp
        }
        
        return global_id

    def mark_lost(self, camera_id: str, local_track_id: int, timestamp: Optional[int] = None) -> None:
        """Mark a local track as lost in specific camera."""
        camera_id_str = str(camera_id)
        local_id = int(local_track_id)
        timestamp = timestamp if timestamp is not None else int(time.time())
        
        # Find global ID
        global_id = self._local_to_global.pop((camera_id_str, local_id), None)
        if global_id is None:
            return
            
        # Update entity state for this camera
        entity = self._global_entities.get(global_id)
        if entity is not None:
            camera_state = entity.get_camera_state(camera_id_str)
            camera_state.mark_lost(timestamp)
            
            # Clean up assignment log
            self._assignment_logs.pop((camera_id_str, local_id), None)
            
            print(f"[EnhancedGlobalIDManager] Global ID {global_id} marked lost in {camera_id_str}")

    def update_coexistence(
        self, 
        camera_id: str, 
        global_ids: Sequence[int], 
        timestamp: Optional[int] = None
    ) -> None:
        """Update coexistence information for global IDs."""
        timestamp = timestamp if timestamp is not None else int(time.time())
        gids = [int(gid) for gid in global_ids if gid is not None]
        
        if len(gids) < 2:
            return
            
        # Record mutual coexistence
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                gid_a, gid_b = gids[i], gids[j]
                
                entity_a = self._global_entities.get(gid_a)
                entity_b = self._global_entities.get(gid_b)
                
                if entity_a is not None:
                    entity_a.update_coexistence(gid_b, timestamp)
                if entity_b is not None:
                    entity_b.update_coexistence(gid_a, timestamp)

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        """Get global ID for local track."""
        return self._local_to_global.get((str(camera_id), int(local_track_id)))

    def get_assignment_info(self, camera_id: str, local_track_id: int) -> Optional[Dict]:
        """Get detailed assignment information."""
        return self._assignment_logs.get((str(camera_id), int(local_track_id)))

    def get_entity_info(self, global_id: int) -> Optional[Dict]:
        """Get detailed information about a global entity."""
        entity = self._global_entities.get(global_id)
        if entity is None:
            return None
            
        return {
            "global_id": entity.global_id,
            "total_observations": entity.total_observations,
            "created_timestamp": entity.created_timestamp,
            "last_updated": entity.last_updated,
            "active_cameras": list(entity.get_active_cameras()),
            "camera_states": {
                cam_id: {
                    "status": state.status.value,
                    "local_id": state.local_id,
                    "last_timestamp": state.last_timestamp,
                    "observations": state.observations
                }
                for cam_id, state in entity.camera_states.items()
            },
            "coexistence_count": len(entity.coexistence_history),
            "feature_history_size": len(entity.feature_history)
        }

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        current_time = int(time.time())
        
        total_entities = len(self._global_entities)
        active_entities = len([e for e in self._global_entities.values() if e.get_active_cameras()])
        total_assignments = len(self._local_to_global)
        
        camera_stats = {}
        for entity in self._global_entities.values():
            for camera_id, state in entity.camera_states.items():
                if camera_id not in camera_stats:
                    camera_stats[camera_id] = {
                        "active": 0, "lost": 0, "inactive": 0, "never_seen": 0
                    }
                camera_stats[camera_id][state.status.value] += 1
        
        return {
            "total_global_entities": total_entities,
            "active_global_entities": active_entities,
            "total_assignments": total_assignments,
            "camera_statistics": camera_stats,
            "next_global_id": self._next_global_id,
            "qdrant_enabled": self.qdrant_bank is not None,
            "current_timestamp": current_time
        }

    def cleanup_inactive_entities(self, force: bool = False) -> int:
        """Clean up old inactive entities and return number cleaned."""
        current_time = int(time.time())
        self._update_entity_states(current_time)
        
        # Find entities that are inactive in all cameras
        to_remove = []
        for global_id, entity in self._global_entities.items():
            all_inactive = True
            for state in entity.camera_states.values():
                if state.status in (CameraStatus.ACTIVE, CameraStatus.LOST):
                    all_inactive = False
                    break
            
            if all_inactive:
                # Check if inactive long enough
                if force or (current_time - entity.last_updated) > self.inactive_timeout_seconds:
                    to_remove.append(global_id)
        
        # Remove inactive entities
        for global_id in to_remove:
            self._global_entities.pop(global_id, None)
            # Clean up any remaining assignments (shouldn't happen)
            keys_to_remove = [
                key for key, gid in self._local_to_global.items() 
                if gid == global_id
            ]
            for key in keys_to_remove:
                self._local_to_global.pop(key, None)
                self._assignment_logs.pop(key, None)
        
        return len(to_remove)

    # Compatibility methods for existing code
    def iter_entities(self):
        """Compatibility method - iterate over global entities."""
        for global_id, entity in self._global_entities.items():
            # Convert to old EntityRecord-like structure for compatibility
            yield global_id, entity

    def get_similarity(self, camera_id: str, local_track_id: int) -> Optional[float]:
        """Get similarity score for assignment."""
        info = self.get_assignment_info(camera_id, local_track_id)
        return info.get("similarity") if info else None


# For backward compatibility, create alias
GlobalIDManager = EnhancedGlobalIDManager
