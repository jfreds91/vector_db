"""Tests for LMDB backend drop_backend() fix."""

import pytest
import tempfile
import shutil
from pathlib import Path

from jfdb.backend.lmdb_backend import LMDBBackend
from jfdb.nodes.node import Node
import torch


class TestDropBackendReset:
    """Tests to verify that drop_backend() properly resets the environment."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for the database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def backend(self, temp_db_path):
        """Create a backend instance with a temporary database."""
        backend = LMDBBackend(name=temp_db_path)
        backend.init_backend()
        yield backend
        # Cleanup
        try:
            if backend.env is not None:
                backend.env.close()
        except:
            pass

    def test_drop_backend_resets_env_to_none(self, backend):
        """Verify that drop_backend() sets self.env to None."""
        assert backend.env is not None
        backend.drop_backend()
        assert backend.env is None

    def test_write_node_after_drop_backend(self, backend):
        """Verify that write_node works after drop_backend() by reinitializing."""
        # Create a sample node
        node = Node(
            id="test_node",
            layers=5,
            embedding=torch.randn(512),
            filepath="/test/path"
        )

        # Write a node
        backend.write_node(node)

        # Drop backend (clears data and resets env to None)
        backend.drop_backend()

        # Verify env is None
        assert backend.env is None

        # Reinitialize and write again (drop_backend's init check should handle this)
        backend.init_backend()
        backend.write_node(node)

        # Read back the node to verify it was written
        read_node = backend.read_node("test_node")
        assert read_node is not None
        assert read_node.id == "test_node"

    def test_read_node_after_drop_backend(self, backend):
        """Verify that read_node works after drop_backend()."""
        # Create and write a node
        node = Node(
            id="read_test_node",
            layers=3,
            embedding=torch.randn(512),
            filepath="/test/read"
        )
        backend.write_node(node)

        # Drop backend
        backend.drop_backend()
        assert backend.env is None

        # Reinitialize backend
        backend.init_backend()

        # Try to read (should return None since we cleared the DB)
        read_node = backend.read_node("read_test_node")
        assert read_node is None

    def test_get_percent_full_after_drop_backend(self, backend):
        """Verify that get_percent_full works after drop_backend()."""
        # Write some data
        node = Node(
            id="capacity_test",
            layers=2,
            embedding=torch.randn(512),
            filepath="/test/capacity"
        )
        backend.write_node(node)

        # Check capacity before drop
        percent_before = backend.get_percent_full()
        assert 0 <= percent_before <= 1

        # Drop backend
        backend.drop_backend()
        assert backend.env is None

        # Reinitialize
        backend.init_backend()

        # Check capacity after drop (should be near 0)
        percent_after = backend.get_percent_full()
        assert 0 <= percent_after <= 1
        # After drop and reinit, should be much lower
        assert percent_after < percent_before

    def test_multiple_drop_backend_calls(self, backend):
        """Verify that drop_backend() can be called multiple times safely."""
        # Write some data
        node = Node(
            id="multi_drop_test",
            layers=1,
            embedding=torch.randn(512),
            filepath="/test/multi"
        )
        backend.write_node(node)

        # First drop
        backend.drop_backend()
        assert backend.env is None

        # Reinitialize
        backend.init_backend()
        assert backend.env is not None

        # Write new data
        node2 = Node(
            id="multi_drop_test_2",
            layers=1,
            embedding=torch.randn(512),
            filepath="/test/multi2"
        )
        backend.write_node(node2)

        # Second drop
        backend.drop_backend()
        assert backend.env is None

        # Reinitialize again
        backend.init_backend()
        assert backend.env is not None

        # Verify we can still use the backend
        read_node = backend.read_node("multi_drop_test_2")
        assert read_node is None  # Data was cleared

    def test_drop_backend_on_uninitialized_backend(self, temp_db_path):
        """Verify that drop_backend() initializes if env is None."""
        backend = LMDBBackend(name=temp_db_path)

        # Before initialization, env should be None
        assert backend.env is None

        # Calling drop_backend should initialize the backend first
        backend.drop_backend()

        # After drop_backend, env should be reset to None
        assert backend.env is None

        # Cleanup
        backend.init_backend()
        backend.env.close()
