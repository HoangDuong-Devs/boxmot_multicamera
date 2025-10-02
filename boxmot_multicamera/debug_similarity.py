#!/usr/bin/env python3
"""Debug script để kiểm tra similarity thực tế của features trong tracking."""

import numpy as np
import torch
import cv2
from boxmot.multicam import GlobalIDManager

# Global variables để store features
stored_features = {}
stored_global_ids = {}

def debug_feature_similarity():
    print("🔍 Debugging Feature Similarity in Real Tracking")
    print("=" * 60)
    
    # Khởi tạo với threshold thấp để bắt được mọi case
    manager = GlobalIDManager(
        sim_threshold=0.3,  # Rất thấp để debug
        historical_sim_threshold=0.3,
        lost_timeout_seconds=600,
        inactive_timeout_seconds=7200,
        use_qdrant=True,
        qdrant_collection="debug_similarity"
    )
    
    print(f"📋 Debug settings:")
    print(f"   - Cost threshold: {manager.cost_threshold}")
    print(f"   - Historical cost threshold: {manager.historical_cost_threshold}")
    
    def mock_assign_with_debug(camera_id, track_id, feature, timestamp):
        """Mock assignment với debug logging."""
        
        # Store feature để so sánh sau
        feature_key = f"{camera_id}:{track_id}"
        stored_features[feature_key] = feature.copy() if feature is not None else None
        
        # Assign global ID
        gid = manager.assign(camera_id, track_id, feature=feature, timestamp=timestamp)
        stored_global_ids[feature_key] = gid
        
        # Get assignment info
        assignment_info = manager.get_assignment_info(camera_id, track_id)
        if assignment_info:
            match_type = assignment_info.get('match_type', 'unknown')
            similarity = assignment_info.get('similarity', 0.0)
            cost = 1.0 - similarity  # Convert similarity back to cost for display
            
            print(f"\n📊 Assignment: {camera_id}:{track_id} -> Global {gid}")
            print(f"   Match type: {match_type}")
            print(f"   Cost: {cost:.6f} (sim: {similarity:.6f})")            # So sánh với features trước đó
            if feature is not None:
                print(f"   Feature stats: min={feature.min():.3f}, max={feature.max():.3f}, norm={np.linalg.norm(feature):.3f}")
                
                # Tìm features tương tự
                for prev_key, prev_feature in stored_features.items():
                    if prev_key != feature_key and prev_feature is not None:
                        # Calculate manual similarity
                        manual_sim = np.dot(feature, prev_feature) / (np.linalg.norm(feature) * np.linalg.norm(prev_feature))
                        prev_gid = stored_global_ids.get(prev_key, 'unknown')
                        
                        if manual_sim > 0.5:  # Only show promising similarities
                            print(f"   📈 Similar to {prev_key} (GID {prev_gid}): {manual_sim:.6f}")
        
        return gid
    
    # Test scenario: track 1 -> lost -> reappear as track 8
    print(f"\n1️⃣ Simulating track 1 appears:")
    feature1 = np.random.randn(512).astype(np.float32)
    feature1 = feature1 / np.linalg.norm(feature1)  # Normalize
    
    gid1 = mock_assign_with_debug("cam0", 1, feature1, 1000)
    
    print(f"\n2️⃣ Track 1 gets lost:")
    manager.mark_lost("cam0", 1, timestamp=2000)
    
    print(f"\n3️⃣ Track 8 appears (should be same person):")
    # Simulate realistic feature variation (5-20% noise)
    for noise_level in [0.05, 0.1, 0.15, 0.2, 0.3]:
        print(f"\n   🧪 Testing with {noise_level*100}% noise level:")
        
        # Add noise to simulate real ReID variation
        noise = np.random.randn(512).astype(np.float32) * noise_level
        feature8 = feature1 + noise
        feature8 = feature8 / np.linalg.norm(feature8)  # Normalize
        
        # Calculate expected similarity
        expected_sim = np.dot(feature1, feature8) / (np.linalg.norm(feature1) * np.linalg.norm(feature8))
        print(f"     Expected similarity: {expected_sim:.6f}")
        
        gid8 = mock_assign_with_debug("cam0", 8, feature8, 3000)
        
        if gid1 == gid8:
            print(f"     ✅ SUCCESS: Global ID reused!")
            break
        else:
            print(f"     ❌ New Global ID created")
    
    return stored_features, stored_global_ids

if __name__ == "__main__":
    features, gids = debug_feature_similarity()
    
    print(f"\n💡 Summary:")
    print(f"   - Total features stored: {len(features)}")
    print(f"   - Unique global IDs: {len(set(gids.values()))}")
    print(f"   - Features: {list(features.keys())}")
    print(f"   - Global IDs: {list(gids.values())}")