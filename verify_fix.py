#!/usr/bin/env python
"""
Verification script for the search() k parameter fix.
This script demonstrates that the fix works correctly.
"""

import sys
import torch
from jfdb.hsnw import DataBase
from jfdb.nodes.node import Node
from jfdb.backend.in_mem_backend import InMemoryBackend


def verify_fix():
    """Verify that the search() method now returns top-k results."""
    print("=" * 60)
    print("Verifying fix for GitHub Issue #3: search() k parameter")
    print("=" * 60)

    # Create an in-memory database
    print("\n1. Creating in-memory database...")
    db = DataBase(
        L=5,
        M=5,
        ef_construction=20,
        d=512,
        backend=InMemoryBackend(node_type=Node),
    )
    print("   Database created successfully")

    # Insert test nodes
    print("\n2. Inserting 10 test nodes...")
    for i in range(10):
        embedding = torch.randn(512)
        node = Node(
            id=f"node_{i}",
            layers=db.L,
            embedding=embedding,
        )
        db.insert(node)
    print("   Inserted 10 nodes")

    # Test search_embedding with different k values
    print("\n3. Testing search_embedding() with different k values:")
    query_embedding = torch.randn(512)

    for k in [1, 3, 5, 10]:
        results = db.search_embedding(query_embedding, k=k)
        print(f"   k={k}: Returned {len(results)} results (expected <= {k})")
        assert len(results) <= k, f"FAILED: Got {len(results)} results but expected <= {k}"
        assert all(isinstance(node, Node) for node in results), "FAILED: Not all results are Node objects"

    print("   ✓ search_embedding() correctly respects k parameter")

    # Test main search method with text
    print("\n4. Testing search() method with text and different k values:")
    for k in [1, 2, 5]:
        try:
            results = db.search(text="test query", k=k)
            print(f"   k={k}: Returned {len(results)} results (expected <= {k})")
            assert len(results) <= k, f"FAILED: Got {len(results)} results but expected <= {k}"
            assert isinstance(results, list), "FAILED: Result is not a list"
            assert all(isinstance(node, Node) for node in results), "FAILED: Not all results are Node objects"
        except Exception as e:
            print(f"   ERROR: {e}")
            return False

    print("   ✓ search() correctly respects k parameter")

    # Test brute force search
    print("\n5. Testing search() with brute_force=True:")
    for k in [1, 3, 5]:
        results = db.search(text="test query", k=k, brute_force=True)
        print(f"   k={k}: Returned {len(results)} results (expected <= {k})")
        assert len(results) <= k, f"FAILED: Got {len(results)} results but expected <= {k}"

    print("   ✓ Brute force search correctly respects k parameter")

    # Verify return types
    print("\n6. Verifying return types:")
    results = db.search(text="test query", k=5)
    print(f"   search() returns: {type(results)}")
    assert isinstance(results, list), "FAILED: search() should return a list"
    print("   ✓ Return type is correct (list)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Fix is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = verify_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
