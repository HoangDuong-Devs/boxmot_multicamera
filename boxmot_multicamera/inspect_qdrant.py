#!/usr/bin/env python3
"""Script để kiểm tra data structure trong Qdrant."""

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def inspect_qdrant_data():
    print("🔍 Inspecting Qdrant Data Structure")
    print("=" * 40)
    
    try:
        client = QdrantClient(host="localhost", port=6333)
        
        # Lấy thông tin collections
        collections = client.get_collections()
        print(f"📋 Available collections: {[c.name for c in collections.collections]}")
        
        # Kiểm tra collection chính
        collection_name = "enhanced_global_reid"
        
        try:
            collection_info = client.get_collection(collection_name)
            print(f"\n📊 Collection '{collection_name}' info:")
            print(f"   - Points count: {collection_info.points_count}")
            print(f"   - Vector size: {collection_info.config.params.vectors.size}")
            print(f"   - Distance metric: {collection_info.config.params.vectors.distance}")
            
            # Lấy một vài points để xem structure
            if collection_info.points_count > 0:
                points = client.scroll(
                    collection_name=collection_name,
                    limit=5,
                    with_payload=True,
                    with_vectors=False
                )
                
                print(f"\n📄 Sample points structure:")
                for i, point in enumerate(points[0]):
                    print(f"   Point {i+1}:")
                    print(f"      ID: {point.id}")
                    print(f"      Payload: {point.payload}")
                    
        except Exception as e:
            print(f"❌ Error accessing collection '{collection_name}': {e}")
            
        # Kiểm tra collection khác nếu có
        for collection in collections.collections:
            if collection.name != collection_name and "reid" in collection.name.lower():
                print(f"\n🔍 Alternative collection: {collection.name}")
                try:
                    alt_info = client.get_collection(collection.name)
                    print(f"   - Points count: {alt_info.points_count}")
                    
                    if alt_info.points_count > 0:
                        alt_points = client.scroll(
                            collection_name=collection.name,
                            limit=2,
                            with_payload=True,
                            with_vectors=False
                        )
                        print(f"   - Sample payload: {alt_points[0][0].payload if alt_points[0] else 'None'}")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {e}")
        print("   💡 Make sure Qdrant server is running on localhost:6333")

if __name__ == "__main__":
    inspect_qdrant_data()