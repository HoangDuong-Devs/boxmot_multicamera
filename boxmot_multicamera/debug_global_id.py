#!/usr/bin/env python3
"""Debug script để kiểm tra GlobalIDManager behavior khi object reappear."""

import numpy as np
from boxmot.multicam import GlobalIDManager

def test_global_id_reuse():
    print("🔍 Testing Global ID Reuse with Qdrant")
    print("=" * 50)
    
    # Khởi tạo với Qdrant enable
    manager = GlobalIDManager(use_qdrant=True)
    
    # In ra cấu hình
    print(f"📋 Qdrant enabled: {hasattr(manager, '_qdrant_bank') and manager._qdrant_bank is not None}")
    print(f"📋 Similarity threshold: {manager.sim_threshold}")
    print(f"📋 Historical similarity threshold: {manager.historical_sim_threshold}")
    
    # Tạo feature giả
    feature1 = np.random.randn(512).astype(np.float32)
    
    print("\n1️⃣ Initial assignment:")
    # Object xuất hiện lần đầu với local_id = 1
    gid1 = manager.assign("cam1", 1, feature=feature1, timestamp=1000)
    print(f"   cam1:1 -> Global ID {gid1}")
    
    # Kiểm tra entity info
    entity_info = manager.get_entity_info(gid1)
    print(f"   Entity info: {entity_info}")
    
    print("\n2️⃣ Mark as lost:")
    # Object biến mất
    manager.mark_lost("cam1", 1, timestamp=2000)
    entity_info = manager.get_entity_info(gid1)
    print(f"   After lost - Entity status: {entity_info['camera_states']['cam1']['status']}")
    
    # Kiểm tra Qdrant storage nếu có
    if hasattr(manager, '_qdrant_bank') and manager._qdrant_bank is not None:
        print(f"   📊 Qdrant has data: {len(manager._qdrant_bank.get_feature_ids()) > 0}")
        feature_ids = manager._qdrant_bank.get_feature_ids()
        print(f"   📊 Feature IDs in Qdrant: {feature_ids}")
    
    print("\n3️⃣ Object reappears with similar feature:")
    # Object xuất hiện lại với local_id = 8 nhưng feature tương tự
    # Thêm một chút noise để test similarity
    feature8 = feature1 + np.random.randn(512).astype(np.float32) * 0.1
    
    gid8 = manager.assign("cam1", 8, feature=feature8, timestamp=3000)
    print(f"   cam1:8 -> Global ID {gid8}")
    
    # Lấy assignment info để xem match type
    assignment_info = manager.get_assignment_info("cam1", 8)
    print(f"   Assignment info: {assignment_info}")
    
    print("\n🎯 Test Result:")
    if gid1 == gid8:
        print(f"   ✅ SUCCESS: Global ID được tái sử dụng ({gid1} == {gid8})")
    else:
        print(f"   ❌ FAIL: Global ID mới được tạo ({gid1} != {gid8})")
        print(f"   🔍 Lý do có thể:")
        print(f"       - Feature similarity quá thấp")
        print(f"       - Qdrant không hoạt động")
        print(f"       - Historical matching bị lỗi")
    
    # Statistics
    try:
        stats = manager.get_statistics()
        print(f"\n📊 Final statistics: {stats}")
    except AttributeError:
        print(f"\n📊 Statistics method not available")

if __name__ == "__main__":
    test_global_id_reuse()