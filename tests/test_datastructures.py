"""Tests for MaxHeap data structure."""

import pytest

from jfdb.utils.datastructures import MaxHeap


class TestMaxHeap:
    """Tests for the MaxHeap class."""

    def test_push_and_pop_returns_highest_priority_first(self) -> None:
        """Verify that pop returns items in descending priority order."""
        heap = MaxHeap()
        heap.push(1, "low")
        heap.push(3, "high")
        heap.push(2, "medium")

        priority, obj = heap.pop()
        assert priority == 3
        assert obj == "high"

        priority, obj = heap.pop()
        assert priority == 2
        assert obj == "medium"

        priority, obj = heap.pop()
        assert priority == 1
        assert obj == "low"
