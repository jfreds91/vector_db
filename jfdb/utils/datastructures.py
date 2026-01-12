import heapq

class MaxHeap:
    """Max heap implementation using Python's min heapq with negated priorities.

    Provides efficient tracking of top-k items by priority (maximum to minimum).
    """

    def __init__(self) -> None:
        """Initialize an empty max heap."""
        self._heap = []

    def push(self, priority: float, obj: any) -> None:
        """Add an item to the heap with a given priority.

        Args:
            priority: Numeric priority value (higher is better).
            obj: Object to store in the heap.
        """
        # Use negative priority to simulate a max heap with heapq (which is a min heap by default)
        heapq.heappush(self._heap, (-priority, obj))

    def pop(self) -> tuple:
        """Remove and return the item with the highest priority.

        Returns:
            Tuple of (priority, obj) with highest priority.
        """
        # Extract the item with the maximum priority (inverting the priority back)
        priority, obj = heapq.heappop(self._heap)
        return -priority, obj

    def peek(self) -> tuple:
        """View the highest priority item without removing it.

        Returns:
            Tuple of (priority, obj) with highest priority, or None if empty.
        """
        # Peek at the maximum priority item without popping it
        if self._heap:
            priority, obj = self._heap[0]
            return -priority, obj
        return None

    def peek_priority(self) -> float:
        """View the highest priority value without removing it.

        Returns:
            The highest priority value, or None if empty.
        """
        # Peek at the maximum priority item without popping it
        if self._heap:
            priority, _ = self._heap[0]
            return -priority
        return None

    def __len__(self) -> int:
        """Return the number of items in the heap."""
        return len(self._heap)

    def dump_to_list(self) -> tuple:
        """Extract all items as sorted lists (highest to lowest priority).

        Returns:
            Tuple of (objects_list, priorities_list) both sorted by priority descending.
        """
        # Dump all elements in the heap to a list, ordered by priority (max to min)
        result_obj = []
        result_priority = []
        while self._heap:
            priority, obj = self.pop()
            result_obj.append(obj)
            result_priority.append(priority)
        return result_obj, result_priority