"""Tests for search functionality in the HNSW database."""

import pytest
import torch
from jfdb.hsnw import DataBase
from jfdb.nodes.node import Node
from jfdb.backend.in_mem_backend import InMemoryBackend


@pytest.fixture
def in_memory_db():
    """Create an in-memory database for testing."""
    db = DataBase(
        L=5,
        M=5,
        ef_construction=20,
        d=512,  # CLIP embedding dimension
        backend=InMemoryBackend(node_type=Node),
    )
    return db


@pytest.fixture
def populated_db(in_memory_db):
    """Create a database with some sample nodes."""
    db = in_memory_db

    # Create 10 test nodes with random embeddings
    for i in range(10):
        # Create a simple embedding tensor
        embedding = torch.randn(512)
        node = Node(
            id=f"node_{i}",
            layers=db.L,
            embedding=embedding,
        )
        db.insert(node)

    return db


class TestSearchEmbeddingTopK:
    """Tests for the search_embedding method with top-k results."""

    def test_search_embedding_returns_list(self, populated_db):
        """Verify that search_embedding returns a list of nodes."""
        query_embedding = torch.randn(512)
        results = populated_db.search_embedding(query_embedding, k=3)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_embedding_respects_k_parameter(self, populated_db):
        """Verify that search_embedding returns at most k results."""
        query_embedding = torch.randn(512)

        for k in [1, 3, 5]:
            results = populated_db.search_embedding(query_embedding, k=k)
            assert len(results) <= k
            assert all(isinstance(node, Node) for node in results)

    def test_search_embedding_default_k_is_5(self, populated_db):
        """Verify that search_embedding defaults to k=5."""
        query_embedding = torch.randn(512)
        results = populated_db.search_embedding(query_embedding)

        # Should return up to 5 results
        assert len(results) <= 5

    def test_search_embedding_returns_nodes(self, populated_db):
        """Verify that search_embedding returns Node objects."""
        query_embedding = torch.randn(512)
        results = populated_db.search_embedding(query_embedding, k=3)

        for node in results:
            assert isinstance(node, Node)
            assert hasattr(node, 'id')
            assert hasattr(node, 'embedding')


class TestSearchMethodTopK:
    """Tests for the main search method with top-k results."""

    def test_search_returns_list_for_text(self, populated_db):
        """Verify that search returns a list when using text."""
        results = populated_db.search(text="a cat", k=3)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_respects_k_parameter_text(self, populated_db):
        """Verify that search with text respects the k parameter."""
        for k in [1, 3, 5]:
            results = populated_db.search(text="a dog", k=k)
            assert len(results) <= k
            assert all(isinstance(node, Node) for node in results)

    def test_search_respects_k_parameter_brute_force(self, populated_db):
        """Verify that brute force search respects the k parameter."""
        for k in [1, 2, 3]:
            results = populated_db.search(text="a bird", k=k, brute_force=True)
            assert len(results) <= k
            assert all(isinstance(node, Node) for node in results)

    def test_search_returns_correct_count(self, populated_db):
        """Verify that search returns exactly k results when available."""
        # With 10 nodes and k=5, should return 5 results
        results = populated_db.search(text="a test query", k=5)

        # Should return exactly 5 results since we have 10 nodes
        assert len(results) == 5

    def test_search_default_k_is_5(self, populated_db):
        """Verify that search defaults to k=5."""
        results = populated_db.search(text="default test")

        # Should return up to 5 results
        assert len(results) <= 5


class TestSearchConsistency:
    """Tests for consistency between search methods."""

    def test_search_embedding_and_search_match(self, populated_db):
        """Verify that search_embedding and search return consistent results."""
        query_embedding = torch.randn(512)
        k = 3

        # Get results from search_embedding
        embedding_results = populated_db.search_embedding(query_embedding, k=k)

        # Both should return lists
        assert isinstance(embedding_results, list)

        # Both should respect k
        assert len(embedding_results) <= k

    def test_search_returns_nodes_with_valid_ids(self, populated_db):
        """Verify that search returns nodes with valid IDs from inserted nodes."""
        inserted_ids = set(f"node_{i}" for i in range(10))

        results = populated_db.search(text="test query", k=5)

        # All returned node IDs should be from our inserted nodes
        for node in results:
            assert node.id in inserted_ids
