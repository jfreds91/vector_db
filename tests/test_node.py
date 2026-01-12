"""Tests for Node class and edge handling."""

import pytest
import logging
import torch
from jfdb.nodes.node import Node


class TestNodeEdges:
    """Tests for Node edge management."""

    def test_add_edge_creates_edge(self) -> None:
        """Verify that adding an edge works correctly."""
        node1 = Node(id="node1", layers=2, embedding=torch.randn(10))
        node2 = Node(id="node2", layers=2, embedding=torch.randn(10))

        node1.add_edge(layer=0, node=node2)

        # Check that edge was added to node1
        assert node2.key in node1.layer_edges[0]
        # Check that reciprocal edge was added to node2
        assert node1.key in node2.layer_edges[0]

    def test_add_duplicate_edge_logs_warning(self, caplog) -> None:
        """Verify that adding a duplicate edge logs a warning instead of raising."""
        node1 = Node(id="node1", layers=2, embedding=torch.randn(10))
        node2 = Node(id="node2", layers=2, embedding=torch.randn(10))

        # Add edge first time
        node1.add_edge(layer=0, node=node2)

        # Add duplicate edge - should not raise, should log warning
        with caplog.at_level(logging.WARNING):
            node1.add_edge(layer=0, node=node2)

        # Verify warning was logged
        assert any("already has an edge" in record.message for record in caplog.records)

        # Verify edge list was not duplicated
        edge_count = node1.layer_edges[0].count(node2.key)
        assert edge_count == 1, f"Expected 1 edge, but found {edge_count}"

    def test_add_duplicate_edge_silently_skips(self) -> None:
        """Verify that adding a duplicate edge doesn't raise KeyError."""
        node1 = Node(id="node1", layers=2, embedding=torch.randn(10))
        node2 = Node(id="node2", layers=2, embedding=torch.randn(10))

        # Add edge first time
        node1.add_edge(layer=0, node=node2)

        # Add duplicate edge - should not raise
        try:
            node1.add_edge(layer=0, node=node2)
        except KeyError:
            pytest.fail("add_edge raised KeyError for duplicate edge")

    def test_add_edge_to_different_layers(self) -> None:
        """Verify that adding edges to different layers works correctly."""
        node1 = Node(id="node1", layers=3, embedding=torch.randn(10))
        node2 = Node(id="node2", layers=3, embedding=torch.randn(10))

        # Add edge to different layers
        node1.add_edge(layer=0, node=node2)
        node1.add_edge(layer=1, node=node2)

        # Both layers should have the edge
        assert node2.key in node1.layer_edges[0]
        assert node2.key in node1.layer_edges[1]

    def test_add_duplicate_edge_different_layers(self) -> None:
        """Verify that same edge can be added to different layers."""
        node1 = Node(id="node1", layers=3, embedding=torch.randn(10))
        node2 = Node(id="node2", layers=3, embedding=torch.randn(10))

        # Add edge to layer 0
        node1.add_edge(layer=0, node=node2)

        # Adding same nodes to different layer should work
        node1.add_edge(layer=1, node=node2)

        # Both layers should have the edge
        assert node2.key in node1.layer_edges[0]
        assert node2.key in node1.layer_edges[1]

    def test_reciprocal_edge_not_added_twice_on_duplicate(self) -> None:
        """Verify that reciprocal edges aren't duplicated when adding duplicate."""
        node1 = Node(id="node1", layers=2, embedding=torch.randn(10))
        node2 = Node(id="node2", layers=2, embedding=torch.randn(10))

        # Add edge first time
        node1.add_edge(layer=0, node=node2)

        # Count edges before attempting duplicate
        reciprocal_edge_count_before = node2.layer_edges[0].count(node1.key)

        # Add duplicate edge
        node1.add_edge(layer=0, node=node2)

        # Count edges after attempting duplicate
        reciprocal_edge_count_after = node2.layer_edges[0].count(node1.key)

        # Reciprocal edge count should not change
        assert reciprocal_edge_count_before == reciprocal_edge_count_after
        assert reciprocal_edge_count_after == 1
