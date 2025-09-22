# assign_id_manager.py

from typing import List

class AssignTree:
    def __init__(self):
        self.assigntree = {}

    def create_link_node(self, child, parent):
        # Guard: tránh loop (child trở thành ancestor của parent)
        if self.search(parent) == child:
            print(f"⚠ Loop detected: cannot link {child} → {parent}")
            return
        self.assigntree[child] = parent

    def search(self, child):
        if child in self.assigntree:
            return self.search(self.assigntree[child])
        return child


class AssignIDManager:
    def __init__(self, max_age=30):
        self.max_age = max_age

        self.local_assignment = {}       # key → (current_id, expected_id, count)
        self.approved_assignment = {}    # key → True/False
        self.tuple_suspect_swapped_id = []
        self.coexistence_ids = {}           
        self.ids = []
        self.assigntree = AssignTree()
        self.removed_keys = []
        self.assign_id_request = []

    def update_attributes(self, **kwargs):
        self.tuple_suspect_swapped_id = kwargs.get("tuple_suspect_swapped_id", self.tuple_suspect_swapped_id)
        self.ids = kwargs.get("ids", self.ids)
        
        coexist = kwargs.get("coexistence_ids", self.coexistence_ids)
        if not isinstance(coexist, dict):
            coexist = {}  # fallback nếu có lỗi
        self.coexistence_ids = coexist

        self.assign_id_request = kwargs.get("assign_id_request", self.assign_id_request)

    def approve_suspect_case_1(self, trackers):
        id_map = {t.id: t for t in trackers}
        for track_id in self.assign_id_request:
            cr_id, expected_id, count = self.local_assignment.get(track_id, (None, None, 0))
            if expected_id is not None and cr_id != expected_id:
                print(f"[AssignCase1] track_id={track_id} | cr_id={cr_id} → expected={expected_id} | count={count}")
                if count > 10:
                    if cr_id > expected_id and expected_id not in self.coexistence_ids.get(cr_id, []):
                        self.approved_assignment[track_id] = True
                        self.assigntree.create_link_node(cr_id, expected_id)
                        print(f"[AssignCase1] ✅ APPROVED: {cr_id} → {expected_id}")
                        if expected_id in self.ids:
                            self.removed_keys.append(expected_id)


    def approve_suspect_case_2(self, trackers):
        id_map = {t.id: t for t in trackers}
        print(f"[AssignCase2] 🔍 tuple_suspect_swapped_id = {self.tuple_suspect_swapped_id}")
        for id1, id2 in self.tuple_suspect_swapped_id:
            t1, t2 = id_map.get(id1), id_map.get(id2)
            if t1 is None or t2 is None:
                print(f"[AssignCase2] ⚠️ Swap case ID not found: {id1}, {id2}")
                continue

            t1.update_major()
            t2.update_major()
            
            major1, _ = t1.suspect_data.get("major", (None, 0))
            major2, _ = t2.suspect_data.get("major", (None, 0))
            print(f"[AssignCase2] Checking pair: (t1.id={t1.id}, major1={major1}) ↔ (t2.id={t2.id}, major2={major2})")

            if t1.id == major2 and t2.id == major1:
                print(f"[AssignCase2] ✅ APPROVED SWAP: {t1.id} ↔ {t2.id}")
                self.approved_assignment[t1.id] = True
                self.approved_assignment[t2.id] = True
                self.local_assignment[t1.id] = (t1.id, major1, 0)
                self.local_assignment[t2.id] = (t2.id, major2, 0)
                t1.reset_suspect()
                t2.reset_suspect()
                
                # Remove tuple để không check lại
                if (id1, id2) in self.tuple_suspect_swapped_id:
                    self.tuple_suspect_swapped_id.remove((id1, id2))


    def approve_suspect_case_3(self, trackers):
        print(f"[AssignCase3] 🔍 Scanning local_assignment...")
        for key, (cr_id, expected_id, count) in self.local_assignment.items():
            print(f"[AssignCase3] key={key} | cr_id={cr_id} → expected_id={expected_id} | count={count}")
            if expected_id is not None and cr_id != expected_id:
                if count > 15:  # Hoặc self.suspect_time * 2
                    if cr_id > expected_id:  # vẫn giữ điều kiện lớn -> nhỏ
                        print(f"[AssignCase3] ✅ APPROVED (Fail-safe): {cr_id} → {expected_id}")
                        self.approved_assignment[key] = True
                        self.assigntree.create_link_node(cr_id, expected_id)
                        if expected_id in self.ids:
                            self.removed_keys.append(expected_id)


    def assign_id(self, trackers):
        id_map = {t.id: t for t in trackers}
        for track_id, approved in self.approved_assignment.items():
            if approved and track_id in self.local_assignment:
                cr_id, expected_id, _ = self.local_assignment[track_id]
                tracker = id_map.get(track_id)
                if tracker and expected_id and expected_id != tracker.id:
                    print(f"[AssignID] ✅ ASSIGN {tracker.id} → {expected_id}")
                    tracker.id = expected_id
                    tracker.reset_suspect(False)

                    # ✅ Reset các track khác có expected_id trùng với ID vừa dùng
                    for k, (cr, exp, cnt) in list(self.local_assignment.items()):
                        if exp == expected_id and k != track_id:
                            print(f"[AssignID] 🔄 Reset {k} vì expected_id bị chiếm: {expected_id}")
                            self.local_assignment[k] = (cr, None, 0)

                    # ✅ Update mapping cho ID mới
                    self.local_assignment[expected_id] = (expected_id, None, 0)
                    if track_id in self.local_assignment:
                        del self.local_assignment[track_id]

                # Reset cờ approved
                self.approved_assignment[track_id] = False

        # Refresh IDs
        self.ids = list(id_map.keys())


    def assign_to_root(self, trackers):
        id_map = {t.id: t for t in trackers}
        for track_id, (curr_id, _, _) in list(self.local_assignment.items()):
            root_id = self.assigntree.search(curr_id)
            if curr_id != root_id and track_id in id_map:
                print(f"[AssignRoot] Tracker {track_id}: {curr_id} → Root: {root_id}")
                tracker = id_map[track_id]
                tracker.id = root_id

                if root_id in self.local_assignment:
                    old_cr, old_exp, old_count = self.local_assignment[root_id]
                    new_count = max(old_count, self.local_assignment[track_id][2])
                    self.local_assignment[root_id] = (root_id, None, new_count)
                else:
                    self.local_assignment[root_id] = (root_id, None, 0)
                # ✅ Update key trong local_assignment
                if track_id in self.local_assignment:
                    del self.local_assignment[track_id]

        # ✅ Refresh ids
        self.ids = list(id_map.keys())

    def update_trk_attributes(self, trackers):
        id_map = {t.id: t for t in trackers}

        for track_id in list(self.local_assignment.keys()):
            if track_id not in id_map:  # Tracker đã bị remove
                # print(f"[UpdateAttr] ❌ Remove {track_id} (tracker removed)")
                del self.local_assignment[track_id]
            else:
                cr_id, expected_id, count = self.local_assignment[track_id]
                if not getattr(id_map[track_id], "suspect", False):
                    if expected_id is not None:
                        print(f"[UpdateAttr] Reset expected_id for {track_id}, keep count={count}")
                    self.local_assignment[track_id] = (cr_id, None, count)

        for track_id in list(self.approved_assignment.keys()):
            if track_id not in id_map:
                print(f"[UpdateAttr] ❌ Remove approved {track_id} (tracker removed)")
                del self.approved_assignment[track_id]

        before_cleanup = len(self.assign_id_request)
        self.assign_id_request = [tid for tid in self.assign_id_request if tid in id_map]
        after_cleanup = len(self.assign_id_request)
        if before_cleanup != after_cleanup:
            print(f"[UpdateAttr] Cleanup assign_id_request: {before_cleanup} → {after_cleanup}")

        for rm_id in self.removed_keys:
            if rm_id in self.local_assignment:
                print(f"[UpdateAttr] 🗑 Remove stale entry {rm_id}")
                del self.local_assignment[rm_id]
        self.removed_keys = []




