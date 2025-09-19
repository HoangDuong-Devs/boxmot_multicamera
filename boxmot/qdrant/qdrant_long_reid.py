# boxmot/qdrant/qdrant_long_reid.py
import numpy as np
import uuid
from uuid import uuid5
from typing import Dict, Tuple, Optional, List

# Qdrant (optional)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, MatchValue, FilterSelector,
        SearchRequest, SearchGroupsRequest
    )
    _HAVE_QDRANT = True
except Exception:
    _HAVE_QDRANT = False


def l2_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v)) + 1e-6
    return (v / n).astype(np.float32)

# ----------------------------
# Local in-memory ring buffer
# ----------------------------
class LocalLongBank:
    """
    (run_uid, track_id) -> ring buffer của size `slots`.
    Mỗi slot lưu (vector, frame_id).
    """
    def __init__(self, dim: int = 512, slots: int = 32):
        self.dim = int(dim)
        self.slots = int(slots)
        # key: (run_uid, track_id, camera_id)
        self._store: Dict[Tuple[str, int, Optional[str]], Dict[int, Tuple[np.ndarray, int]]] = {}

    def _key(self, run_uid: str, track_id: int, camera_id: Optional[str]) -> Tuple[str, int, Optional[str]]:
        cam = str(camera_id) if camera_id is not None else None
        return (run_uid, int(track_id), cam)

    def add_ring(self, run_uid: str, track_id: int, vector: np.ndarray,
                 slot: int, extra: Optional[dict] = None,
                 camera_id: Optional[str] = None) -> str:
        key = self._key(run_uid, track_id, camera_id)
        vec = l2_norm(vector)
        slot = int(slot) % self.slots
        if key not in self._store:
            self._store[key] = {}
        frame_id = int(extra.get("frame_id", 0)) if extra else 0
        self._store[key][slot] = (vec, frame_id)
        # pseudo id
        cam_tag = str(camera_id) if camera_id is not None else "-"
        return f"local:{run_uid}:{cam_tag}:{int(track_id)}:{slot}"

    def delete_track(self, run_uid: str, track_id: int, camera_id: Optional[str] = None):
        self._store.pop(self._key(run_uid, track_id, camera_id), None)

    def get_slot_vector(self, run_uid: str, track_id: int, slot: int,
                        camera_id: Optional[str] = None):
        """Lấy đúng vector và frame theo slot."""
        key = self._key(run_uid, track_id, camera_id)
        d = self._store.get(key)
        if not d:
            return None, None
        slot = int(slot) % self.slots
        if slot not in d:
            return None, None
        vec, fr = d[slot]
        return vec.copy(), int(fr)

    def get_all_vectors(self, run_uid: str, track_id: int, limit: int = 100,
                         camera_id: Optional[str] = None):
        key = self._key(run_uid, track_id, camera_id)
        d = self._store.get(key)
        if not d:
            return None
        # sort theo frame_id để ổn định
        items = sorted(d.items(), key=lambda kv: kv[1][1])
        vecs = [it[1][0] for it in items][:int(limit)]
        return np.stack(vecs, axis=0) if vecs else None

    def get_vectors_with_payload(self, run_uid: str, track_id: int,
                                  page_size: int = 128,
                                  camera_id: Optional[str] = None):
        """Giữ nguyên signature cũ: trả (vecs, frames)."""
        key = self._key(run_uid, track_id, camera_id)
        d = self._store.get(key)
        if not d:
            return None, None
        items = sorted(d.items(), key=lambda kv: kv[1][1])
        vecs   = [it[1][0] for it in items]
        frames = [it[1][1] for it in items]
        if not vecs:
            return None, None
        return np.stack(vecs, axis=0), np.asarray(frames, dtype=np.int64)

    def get_vectors_frames_slots(self, run_uid: str, track_id: int,
                                  camera_id: Optional[str] = None):
        """Mới: trả (vecs, frames, slots) đã sắp theo frame tăng dần."""
        key = self._key(run_uid, track_id, camera_id)
        d = self._store.get(key)
        if not d:
            return None, None, None
        items = sorted(d.items(), key=lambda kv: kv[1][1])
        slots  = [it[0] for it in items]
        vecs   = [it[1][0] for it in items]
        frames = [it[1][1] for it in items]
        if not vecs:
            return None, None, None
        return np.stack(vecs, axis=0), np.asarray(frames, dtype=np.int64), np.asarray(slots, dtype=np.int32)

    def centroid(self, run_uid: str, track_id: int, camera_id: Optional[str] = None):
        """Trung bình cộng (mean) các vector hiện có của track."""
        V = self.get_all_vectors(run_uid, track_id, camera_id=camera_id)
        if V is None or len(V) == 0:
            return None
        c = V.mean(axis=0)
        return l2_norm(c)


# ----------------------------
# Qdrant remote ring buffer
# ----------------------------
class QdrantLongBank:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "long_term_reid",
        dim: int = 512,
        slots: int = 32,
    ):
        if not _HAVE_QDRANT:
            raise RuntimeError("qdrant-client not installed")
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        self.dim = int(dim)
        self.slots = int(slots)
        self._uuid_ns = uuid.uuid5(uuid.NAMESPACE_DNS, "boxmot.long_term_reid")
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def _recreate_with_dim(self, new_dim: int):
        self.dim = int(new_dim)
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    # ---- helpers ----
    def _point_id(
        self,
        run_uid: str,
        track_id: int,
        slot: int,
        camera_id: Optional[str] = None,
    ) -> str:
        cam = str(camera_id) if camera_id is not None else "-"
        name = f"{run_uid}:{cam}:{int(track_id)}:{int(slot) % self.slots}"
        return str(uuid5(self._uuid_ns, name))

    # ---- API ----
    def add_ring(
        self,
        run_uid: str,
        track_id: int,
        vector: np.ndarray,
        slot: int,
        extra: Optional[dict] = None,
        camera_id: Optional[str] = None,
    ) -> str:
        """Upsert theo ring-buffer (id theo run_uid/track_id/slot)."""
        point_id = self._point_id(run_uid, track_id, slot, camera_id)
        payload = {
            "run_uid": run_uid,
            "track_id": int(track_id),
            "slot": int(slot) % self.slots,
        }
        if camera_id is not None:
            payload["camera_id"] = str(camera_id)
        if extra:
            payload.update(extra)

        v = l2_norm(vector)
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=point_id, vector=v.tolist(), payload=payload)],
            )
        except Exception as e:
            msg = str(e).lower()
            # auto-fix wrong vector size by recreating with the correct dim
            if any(k in msg for k in ["vector size", "incompatible", "wrong"]):
                try:
                    self._recreate_with_dim(len(v))
                    self.client.upsert(
                        collection_name=self.collection,
                        points=[PointStruct(id=point_id, vector=v.tolist(), payload=payload)],
                    )
                except Exception as ee:
                    print(f"[QdrantLongBank.add_ring] recreate+upsert FAILED: {ee} | id={point_id}")
                    raise
            else:
                print(f"[QdrantLongBank.add_ring] upsert FAILED: {e} | id={point_id}")
                raise
        return point_id

    def delete_track(self, run_uid: str, track_id: int, camera_id: Optional[str] = None):
        must = [
            FieldCondition(key="run_uid", match=MatchValue(value=run_uid)),
            FieldCondition(key="track_id", match=MatchValue(value=int(track_id))),
        ]
        if camera_id is not None:
            must.append(FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id))))
        flt = Filter(must=must)
        self.client.delete(collection_name=self.collection, points_selector=FilterSelector(filter=flt))

    def get_slot_vector(
        self,
        run_uid: str,
        track_id: int,
        slot: int,
        camera_id: Optional[str] = None,
    ):
        """Lấy đúng vector và frame theo slot bằng retrieve(id)."""
        pid = self._point_id(run_uid, track_id, slot, camera_id)
        try:
            pts = self.client.retrieve(
                collection_name=self.collection,
                ids=[pid], with_payload=True, with_vectors=True
            )
            if not pts:
                return None, None
            v  = np.asarray(pts[0].vector, dtype=np.float32)
            fr = int(pts[0].payload.get("frame_id", 0))
            return v, fr
        except Exception:
            return None, None

    def get_all_vectors(
        self,
        run_uid: str,
        track_id: int,
        limit: int = 100,
        camera_id: Optional[str] = None,
    ):
        """Lấy tối đa `limit` vector của track (không đảm bảo đủ tất cả slot nếu > limit)."""
        must = [
            FieldCondition(key="run_uid", match=MatchValue(value=run_uid)),
            FieldCondition(key="track_id", match=MatchValue(value=int(track_id))),
        ]
        if camera_id is not None:
            must.append(FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id))))
        flt = Filter(must=must)
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=flt, with_vectors=True, limit=int(limit)
        )
        if not points:
            return None
        return np.stack([np.asarray(p.vector, dtype=np.float32) for p in points], axis=0)

    def get_vectors_with_payload(
        self,
        run_uid: str,
        track_id: int,
        page_size: int = 128,
        camera_id: Optional[str] = None,
    ):
        """
        Giữ nguyên signature cũ: trả (vecs, frames).
        Gợi ý: dùng get_vectors_frames_slots() nếu cần cả slots.
        """
        must = [
            FieldCondition(key="run_uid", match=MatchValue(value=run_uid)),
            FieldCondition(key="track_id", match=MatchValue(value=int(track_id))),
        ]
        if camera_id is not None:
            must.append(FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id))))
        flt = Filter(must=must)
        offset = None
        vecs: List[np.ndarray] = []
        frames: List[int] = []
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt, with_vectors=True,
                limit=int(page_size), offset=offset
            )
            if not points:
                break
            for p in points:
                vecs.append(np.asarray(p.vector, dtype=np.float32))
                frames.append(int(p.payload.get("frame_id", 0)))
            if offset is None:
                break
        if not vecs:
            return None, None
        # sắp theo frame tăng dần cho ổn định
        order = np.argsort(np.asarray(frames, dtype=np.int64))
        V = np.stack(vecs, axis=0)[order]
        F = np.asarray(frames, dtype=np.int64)[order]
        return V, F

    def get_vectors_frames_slots(
        self,
        run_uid: str,
        track_id: int,
        page_size: int = 128,
        camera_id: Optional[str] = None,
    ):
        """
        Mới: trả (vecs, frames, slots), sắp theo frame tăng dần.
        """
        must = [
            FieldCondition(key="run_uid", match=MatchValue(value=run_uid)),
            FieldCondition(key="track_id", match=MatchValue(value=int(track_id))),
        ]
        if camera_id is not None:
            must.append(FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id))))
        flt = Filter(must=must)
        offset = None
        vecs: List[np.ndarray] = []
        frames: List[int] = []
        slots: List[int] = []
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt, with_vectors=True,
                limit=int(page_size), offset=offset
            )
            if not points:
                break
            for p in points:
                vecs.append(np.asarray(p.vector, dtype=np.float32))
                frames.append(int(p.payload.get("frame_id", 0)))
                slots.append(int(p.payload.get("slot", -1)))
            if offset is None:
                break
        if not vecs:
            return None, None, None
        order = np.argsort(np.asarray(frames, dtype=np.int64))
        V = np.stack(vecs, axis=0)[order]
        F = np.asarray(frames, dtype=np.int64)[order]
        S = np.asarray(slots , dtype=np.int32)[order]
        return V, F, S

    def centroid(self, run_uid: str, track_id: int, camera_id: Optional[str] = None):
        """Trung bình cộng (mean) các vector hiện có của track."""
        V, F = self.get_vectors_with_payload(run_uid, track_id, camera_id=camera_id)
        if V is None or len(V) == 0:
            return None
        c = V.mean(axis=0)
        return l2_norm(c)

    def _filter_for(
        self,
        run_uid: str,
        track_ids: Optional[List[int]] = None,
        camera_id: Optional[str] = None,
    ) -> Filter:
        must = [FieldCondition(key="run_uid", match=MatchValue(value=run_uid))]
        if camera_id is not None:
            must.append(FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id))))
        if track_ids:
            # many "should" = IN list
            should = [FieldCondition(key="track_id", match=MatchValue(value=int(t))) for t in track_ids]
            return Filter(must=must, should=should)
        return Filter(must=must)
    
    def grouped_topk_mean(
        self,
        run_uid: str,
        track_ids: List[int],
        queries: np.ndarray,
        k: int = 5,
        camera_id: Optional[str] = None,
    ):
        """
        Server-side: với mỗi query vector, trả dict {track_id: mean_topk_cos}.
        Ưu tiên search_groups (group_by track_id); nếu không có thì fallback search_batch per-track.
        """
        if queries is None or len(queries) == 0:
            return []
        Q = np.asarray(queries, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]

        flt = self._filter_for(run_uid, track_ids, camera_id=camera_id)

        # ---------- try search_groups (best) ----------
        try:
            out = []
            for q in Q:
                res = self.client.search_groups(
                    collection_name=self.collection,
                    query_vector=q.tolist(),
                    group_by="track_id",
                    group_size=int(k),
                    limit=max(1, len(track_ids) if track_ids else 1),  # cố gắng cover tất cả groups
                    with_payload=False,
                    query_filter=flt,
                )
                m = {}
                for g in getattr(res, "groups", []):
                    try:
                        tid = int(g.id)
                    except Exception:
                        continue
                    sims = [h.score for h in (g.hits or [])]
                    if sims:
                        m[tid] = float(sum(sims) / len(sims))
                out.append(m)
            return out
        except Exception as e:
            print(f"[QdrantLongBank.grouped_topk_mean] search_groups failed → fallback: {e}")

        # ---------- fallback: search_batch per-track ----------
        try:
            out = []
            # dựng requests cho 1 query: mỗi track_id là 1 request (lọc track_id riêng)
            def batch_for_query(q):
                reqs = []
                if track_ids:
                    for tid in track_ids:
                        flt_tid = self._filter_for(run_uid, [tid], camera_id=camera_id)
                        reqs.append(SearchRequest(vector=q.tolist(), filter=flt_tid, limit=int(k), with_payload=False))
                else:
                    # không có danh sách → 1 nhóm chung, sau đó map track_id từ payload (ở đây không trả payload → khó)
                    # nên nếu không có track_ids, đành về client-side
                    return None
                return reqs

            for q in Q:
                reqs = batch_for_query(q)
                if reqs is None:
                    out.append({})
                    continue
                results = self.client.search_batch(collection_name=self.collection, requests=reqs)
                m = {}
                for tid, hits in zip(track_ids, results):
                    sims = [h.score for h in (hits or [])]
                    if sims:
                        m[int(tid)] = float(sum(sims) / len(sims))
                out.append(m)
            return out
        except Exception as e:
            print(f"[QdrantLongBank.grouped_topk_mean] search_batch failed: {e}")
            return None  # để caller tự fallback client-side
    
# ----------------------------
# Hybrid adapter
# ----------------------------
class HybridLongBank:
    """
    Ưu tiên Qdrant (nếu bật & có), nếu lỗi thì fallback LocalLongBank.
    API: add_ring, delete_track, centroid, get_all_vectors,
         get_vectors_with_payload, get_slot_vector, get_vectors_frames_slots
    """
    def __init__(
        self,
        use_qdrant: bool = True,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "long_term_reid",
        dim: int = 512,
        slots: int = 32,
    ):
        self.slots = int(slots)
        if use_qdrant and _HAVE_QDRANT:
            try:
                self.backend = QdrantLongBank(host=host, port=port, collection=collection, dim=dim, slots=slots)
                self.backend_name = "qdrant"
            except Exception as e:
                print(f"[HybridLongBank] Qdrant init failed, fallback to Local. Reason: {e}")
                self.backend = LocalLongBank(dim=dim, slots=slots)
                self.backend_name = "local"
        else:
            self.backend = LocalLongBank(dim=dim, slots=slots)
            self.backend_name = "local"

    # proxy
    def add_ring(self, *args, **kwargs): return self.backend.add_ring(*args, **kwargs)
    def delete_track(self, *args, **kwargs): return self.backend.delete_track(*args, **kwargs)
    def centroid(self, *args, **kwargs): return self.backend.centroid(*args, **kwargs)
    def get_all_vectors(self, *args, **kwargs): return self.backend.get_all_vectors(*args, **kwargs)
    def get_vectors_with_payload(self, *args, **kwargs): return self.backend.get_vectors_with_payload(*args, **kwargs)
    def get_slot_vector(self, *args, **kwargs): return self.backend.get_slot_vector(*args, **kwargs)
    def get_vectors_frames_slots(self, *args, **kwargs): return self.backend.get_vectors_frames_slots(*args, **kwargs)
    def grouped_topk_mean(self, *args, **kwargs):
        if getattr(self, "backend_name", "local") != "qdrant":
            return None
        try:
            return self.backend.grouped_topk_mean(*args, **kwargs)
        except Exception as e:
            print(f"[HybridLongBank.grouped_topk_mean] failed: {e}")
            return None
