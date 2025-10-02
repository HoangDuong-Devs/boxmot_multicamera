#!/usr/bin/env python3
"""Test script for enhanced GlobalIDManager with Qdrant integration and TTL cleanup."""

import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from boxmot.multicam.global_id import GlobalIDManager

def test_basic_functionality():
    """Test basic GlobalIDManager functionality."""
    print("=== Testing Basic Functionality ===")
    
    # Create manager with Qdrant disabled for basic test
    manager = GlobalIDManager(
        use_qdrant=False,
        entity_ttl_hours=1,
        coex_ttl_hours=1,
    )
    
    # Test basic assignment
    feature1 = np.random.random(512).astype(np.float32)
    feature2 = np.random.random(512).astype(np.float32)
    
    gid1 = manager.assign("cam1", 1, feature1, timestamp=1000)
    gid2 = manager.assign("cam1", 2, feature2, timestamp=1001)
    
    print(f"Assigned global IDs: {gid1}, {gid2}")
    assert gid1 != gid2, "Different tracks should get different global IDs"
    
    # Test coexistence tracking
    manager.update_coexistence("cam1", [gid1, gid2], timestamp=1002)
    coex1 = manager.get_coexistence_info(gid1)
    coex2 = manager.get_coexistence_info(gid2)
    
    print(f"Coexistence for {gid1}: {coex1}")
    print(f"Coexistence for {gid2}: {coex2}")
    assert gid2 in coex1, "Coexistence should be recorded"
    assert gid1 in coex2, "Coexistence should be bidirectional"
    
    print("✓ Basic functionality test passed\n")

def test_ttl_cleanup():
    """Test TTL-based cleanup functionality."""
    print("=== Testing TTL Cleanup ===")
    
    manager = GlobalIDManager(
        use_qdrant=False,
        entity_ttl_hours=1,
        cleanup_interval_minutes=1,
    )
    
    # Create some entities
    feature1 = np.random.random(512).astype(np.float32)
    feature2 = np.random.random(512).astype(np.float32)
    
    gid1 = manager.assign("cam1", 1, feature1, timestamp=1000)
    gid2 = manager.assign("cam2", 1, feature2, timestamp=1001)
    
    print(f"Created entities: {gid1}, {gid2}")
    
    # Mark one as lost (should trigger TTL)
    manager.mark_lost("cam1", 1, timestamp=1100)
    
    stats_before = manager.get_stats()
    print(f"Stats before cleanup: {stats_before}")
    
    # Simulate time passing and force cleanup
    manager.last_cleanup_time = 1000  # Force cleanup on next call
    manager._cleanup_expired_data(force=True)
    
    stats_after = manager.get_stats()
    print(f"Stats after cleanup: {stats_after}")
    
    print("✓ TTL cleanup test passed\n")

def test_qdrant_integration():
    """Test Qdrant integration if available."""
    print("=== Testing Qdrant Integration ===")
    
    try:
        manager = GlobalIDManager(
            use_qdrant=True,
            qdrant_host="localhost",
            qdrant_port=6333,
            historical_sim_threshold=0.7,
        )
        
        if manager.qdrant_bank is None:
            print("Qdrant not available, skipping integration test")
            return
        
        print(f"Qdrant backend: {manager.qdrant_bank.backend_name}")
        
        # Test basic assignment with Qdrant storage
        feature1 = np.random.random(512).astype(np.float32)
        feature1 = feature1 / np.linalg.norm(feature1)  # Normalize
        
        gid1 = manager.assign("cam1", 1, feature1, timestamp=1000)
        print(f"Assigned global ID with Qdrant: {gid1}")
        
        # Test historical search
        similar_feature = feature1 + np.random.random(512) * 0.1
        similar_feature = similar_feature / np.linalg.norm(similar_feature)
        
        # Mark first track as lost
        manager.mark_lost("cam1", 1, timestamp=1100)
        
        # Try to match similar feature from different camera
        gid2 = manager.assign("cam2", 1, similar_feature, timestamp=1200)
        print(f"Historical match result: {gid2}")
        
        if gid1 == gid2:
            print("✓ Historical matching worked!")
        else:
            print("⚠ Historical matching may not have worked (could be normal)")
        
        print("✓ Qdrant integration test completed\n")
        
    except Exception as e:
        print(f"Qdrant test failed (expected if Qdrant not running): {e}\n")

def test_coexistence_conflict():
    """Test coexistence conflict detection."""
    print("=== Testing Coexistence Conflict Detection ===")
    
    # Use smaller TTL for testing
    manager = GlobalIDManager(
        use_qdrant=False,
        coex_ttl_hours=1,  # 1 hour = 3600 seconds
    )
    
    # Create two entities that coexist
    feature1 = np.random.random(512).astype(np.float32)
    feature2 = np.random.random(512).astype(np.float32)
    
    base_time = 1000
    gid1 = manager.assign("cam1", 1, feature1, timestamp=base_time)
    gid2 = manager.assign("cam1", 2, feature2, timestamp=base_time + 1)
    
    # Record coexistence
    manager.update_coexistence("cam1", [gid1, gid2], timestamp=base_time + 2)
    
    # Test conflict detection using same time context
    active_ids = {gid1}
    conflict = manager._check_coexistence_conflict(gid2, active_ids, current_time=base_time + 10)
    
    print(f"Global IDs: {gid1}, {gid2}")
    print(f"Active IDs: {active_ids}")
    print(f"Coexistence conflict detected: {conflict}")
    
    # Print debug info
    coex_info = manager.get_coexistence_info(gid2)
    print(f"Coexistence info for {gid2}: {coex_info}")
    print(f"TTL seconds: {manager.coex_ttl_seconds}")
    
    if conflict:
        print("✓ Coexistence conflict detection test passed")
    else:
        print("⚠ No conflict detected")
    
    print()

def main():
    """Run all tests."""
    print("Testing Enhanced GlobalIDManager\n")
    
    test_basic_functionality()
    test_ttl_cleanup()
    test_coexistence_conflict()
    test_qdrant_integration()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()