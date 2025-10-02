#!/usr/bin/env python3
"""Script để debug tracking thực tế với enhanced logging."""

import sys
from pathlib import Path
import numpy as np

# Add để import được tracking modules
sys.path.insert(0, str(Path(__file__).parent))

from boxmot.multicam import GlobalIDManager

def debug_real_tracking():
    print("🔍 Debugging Real Tracking Scenario")
    print("=" * 50)
    
    # Khởi tạo như trong tracking file
    manager = GlobalIDManager(
        use_qdrant=True,
        qdrant_collection="debug_real_tracking",
        sim_threshold=0.6,
        historical_sim_threshold=0.7,
        lost_timeout_seconds=300,  # 5 minutes
        inactive_timeout_seconds=3600,  # 1 hour
    )
    
    print(f"📋 Manager settings:")
    print(f"   - Similarity threshold: {manager.sim_threshold}")
    print(f"   - Historical threshold: {manager.historical_sim_threshold}")
    print(f"   - Lost timeout: {manager.lost_timeout_seconds}s")
    print(f"   - Inactive timeout: {manager.inactive_timeout_seconds}s")
    
    # Simulate scenario: object với ID 1 → lost → reappear với ID 8
    print(f"\n1️⃣ Object appears as track ID 1:")
    
    # Tạo feature ngẫu nhiên để mô phỏng
    feature1 = np.random.randn(512).astype(np.float32)
    gid1 = manager.assign("cam0", 1, feature=feature1, timestamp=1000)
    print(f"   cam0:1 -> Global ID {gid1}")
    
    # Simulate một số frames tracking
    for i in range(5):
        # Feature thay đổi nhẹ qua các frame (normal variation)
        varied_feature = feature1 + np.random.randn(512).astype(np.float32) * 0.05
        gid = manager.assign("cam0", 1, feature=varied_feature, timestamp=1000 + i*33)
        print(f"   Frame {i+1}: cam0:1 -> Global ID {gid}")
    
    print(f"\n2️⃣ Object lost:")
    # Object bị lost
    manager.mark_lost("cam0", 1, timestamp=1200)
    entity_info = manager.get_entity_info(gid1)
    print(f"   Status after lost: {entity_info['camera_states']['cam0']['status']}")
    
    print(f"\n3️⃣ Time passes (simulating real delay)...")
    # Simulate thời gian trôi qua - đây có thể là lý do
    # Trong video thực, delay có thể khiến object chuyển từ lost -> inactive
    
    print(f"\n4️⃣ Object reappears as track ID 8:")
    
    # Test với different levels of feature similarity
    similarities = [0.95, 0.8, 0.7, 0.6, 0.5, 0.4]
    
    for sim_level in similarities:
        print(f"\n   🧪 Testing similarity level: {sim_level}")
        
        # Tạo feature với mức độ tương đồng khác nhau
        noise_level = (1.0 - sim_level) * 2.0  # Higher noise for lower similarity
        feature8 = feature1 + np.random.randn(512).astype(np.float32) * noise_level
        
        # Normalize để đảm bảo similarity trong khoảng mong muốn
        # Calculate actual similarity
        actual_sim = np.dot(feature1, feature8) / (np.linalg.norm(feature1) * np.linalg.norm(feature8))
        print(f"     Actual feature similarity: {actual_sim:.3f}")
        
        gid8 = manager.assign("cam0", 8, feature=feature8, timestamp=2000)
        assignment_info = manager.get_assignment_info("cam0", 8)
        
        print(f"     cam0:8 -> Global ID {gid8}")
        print(f"     Match type: {assignment_info.get('match_type', 'unknown')}")
        print(f"     Assignment similarity: {assignment_info.get('similarity', 'unknown')}")
        
        if gid1 == gid8:
            print(f"     ✅ SUCCESS: Global ID reused!")
            break
        else:
            print(f"     ❌ New Global ID created")
    
    print(f"\n💡 Possible reasons for new Global ID in real tracking:")
    print(f"   1. Feature similarity below threshold ({manager.sim_threshold})")
    print(f"   2. Object lost too long (> {manager.lost_timeout_seconds}s)")
    print(f"   3. Camera ID mismatch")
    print(f"   4. ReID model produces very different features")
    print(f"   5. Person appearance changed significantly")

if __name__ == "__main__":
    debug_real_tracking()