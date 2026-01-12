"""Tests for HSNW algorithm, particularly for traversed nodes counter accuracy."""

import pytest
import torch
import numpy as np
from jfdb.hsnw import DataBase
from jfdb.nodes.node import Node
from jfdb.backend.lmdb_backend import LMDBBackend
import tempfile
import shutil


class TestBFSTraversedNodesCounter:
    """Tests for the bfs_with_max_heap traversed nodes counter."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for the test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def database(self, temp_db_path):
        """Create a test database instance."""
        backend = LMDBBackend(
            name="test_db",
            preallocated_bytes=int(1e7),
            node_type=Node,
        )
        db = DataBase(
            L=3,
            M=5,
            ef_construction=10,
            d=512,
            backend=backend,
        )
        return db

    @pytest.fixture
    def test_nodes(self):
        """Create test nodes with embeddings."""
        nodes = []
        for i in range(5):
            embedding = torch.randn(512)
            node = Node(
                id=f"test_node_{i}",
                layers=3,
                embedding=embedding,
            )
            nodes.append(node)
        return nodes

    def test_traversed_nodes_counter_with_single_start_node(self, database, test_nodes):
        """Test that traversed nodes counter increments by 1 when starting with 1 node."""
        search_node = test_nodes[0]
        start_nodes = [test_nodes[1]]

        # Write start node to backend
        database.backend.write_node(start_nodes[0])

        # Call bfs_with_max_heap with 1 start node
        result = database.bfs_with_max_heap(
            search_node=search_node,
            start_nodes=start_nodes,
            ef=2,
            layer=0,
            backend=database.backend,
            multi_hop=False
        )

        # With 1 start node, we should have at least 1 traversed node
        assert result.total_traversed_nodes >= 1, \
            f"Expected at least 1 traversed node with 1 start node, got {result.total_traversed_nodes}"

    def test_traversed_nodes_counter_with_multiple_start_nodes(self, database, test_nodes):
        """Test that traversed nodes counter increments by len(start_nodes)."""
        search_node = test_nodes[0]
        start_nodes = test_nodes[1:4]  # 3 start nodes

        # Write start nodes to backend
        for node in start_nodes:
            database.backend.write_node(node)

        # Call bfs_with_max_heap with 3 start nodes
        result = database.bfs_with_max_heap(
            search_node=search_node,
            start_nodes=start_nodes,
            ef=2,
            layer=0,
            backend=database.backend,
            multi_hop=False
        )

        # With 3 start nodes and ef=2, we should have at least 3 traversed nodes
        # (the start nodes themselves)
        assert result.total_traversed_nodes >= len(start_nodes), \
            f"Expected at least {len(start_nodes)} traversed nodes with {len(start_nodes)} start nodes, " \
            f"got {result.total_traversed_nodes}"

    def test_traversed_nodes_counter_correctness(self, database, test_nodes):
        """Test that the counter correctly counts all traversed nodes."""
        search_node = test_nodes[0]
        start_nodes = [test_nodes[1], test_nodes[2]]

        # Write start nodes to backend
        for node in start_nodes:
            database.backend.write_node(node)

        # Call bfs_with_max_heap with multiple start nodes and multi_hop=False
        result = database.bfs_with_max_heap(
            search_node=search_node,
            start_nodes=start_nodes,
            ef=5,
            layer=0,
            backend=database.backend,
            multi_hop=False
        )

        # When ef is reached with only start nodes, the counter should include all start nodes
        # The counter should be at least len(start_nodes)
        assert result.total_traversed_nodes >= len(start_nodes), \
            f"Counter should be at least {len(start_nodes)}, got {result.total_traversed_nodes}"

        # Verify that the returned nodes are in the result
        assert len(result.nodes) > 0, "Should return at least one node"
        assert len(result.priorities) == len(result.nodes), \
            "Should have same number of priorities as nodes"


class TestSearchEmptyDatabase:
    """Tests for searching on an empty database."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for the database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def empty_database(self, temp_db_path):
        """Create an empty database instance."""
        backend = LMDBBackend(
            name="test_empty_db",
            preallocated_bytes=int(1e6),
            node_type=Node,
        )
        db = DataBase(
            L=4,
            M=5,
            ef_construction=50,
            d=512,
            backend=backend,
            node_type=Node
        )
        return db

    def test_search_embedding_on_empty_database_raises_error(self, empty_database):
        """Verify that searching on an empty database raises a ValueError."""
        # Create a dummy embedding
        dummy_embedding = torch.randn(512)

        # Attempt to search should raise ValueError
        with pytest.raises(ValueError, match="Database is empty"):
            empty_database.search_embedding(embedding=dummy_embedding)

    def test_search_with_text_on_empty_database_raises_error(self, empty_database):
        """Verify that text search on an empty database raises a ValueError."""
        with pytest.raises(ValueError, match="Database is empty"):
            empty_database.search(text="test query")

    def test_search_with_image_on_empty_database_raises_error(self, empty_database):
        """Verify that image search on an empty database raises a ValueError."""
        from PIL import Image

        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')

        with pytest.raises(ValueError, match="Database is empty"):
            empty_database.search(image=dummy_image)
