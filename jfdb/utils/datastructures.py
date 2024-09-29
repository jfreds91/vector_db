import heapq

class MaxHeap:
    def __init__(self):
        self._heap = []

    def push(self, priority, obj):
        # Use negative priority to simulate a max heap with heapq (which is a min heap by default)
        heapq.heappush(self._heap, (-priority, obj))

    def pop(self):
        # Extract the item with the maximum priority (inverting the priority back)
        priority, obj = heapq.heappop(self._heap)
        return -priority, obj

    def peek(self):
        # Peek at the maximum priority item without popping it
        if self._heap:
            priority, obj = self._heap[0]
            return -priority, obj
        return None

    def peek_priority(self):
        # Peek at the maximum priority item without popping it
        if self._heap:
            priority, _ = self._heap[0]
            return -priority
        return None

    def __len__(self):
        # Return the number of items in the heap
        return len(self._heap)

    def dump_to_list(self):
        # Dump all elements in the heap to a list, ordered by priority (max to min)
        result_obj = []
        result_priority = []
        while self._heap:
            priority, obj = self.pop()
            result_obj.append(obj)
            result_priority.append(priority)
        return result_obj, result_priority