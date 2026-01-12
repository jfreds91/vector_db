from jfdb.backend.backend import Backend
from jfdb.nodes.node import Node
from typing import Union, Dict, Optional
import logging


class InMemoryBackend(Backend):
    """Simple in-memory backend for testing purposes."""

    def __init__(self, **kwargs) -> None:
        """Initialize the in-memory backend.

        Args:
            **kwargs: Additional arguments passed to parent Backend class.
        """
        super().__init__(**kwargs)
        self._store: Dict[Union[str, bytes], Node] = {}
        self.env = None  # For compatibility with code expecting env attribute

    def init_backend(self) -> None:
        """Initialize the in-memory backend storage.

        Clears any existing data and prepares the backend for operations.
        """
        self._store = {}
        logging.info('Initialized in-memory backend')

    def drop_backend(self) -> None:
        """Clear all data from the in-memory backend.

        Removes all stored nodes.
        """
        self._store.clear()
        logging.warning('Cleared in-memory backend')

    def write_node(self, node: Node) -> None:
        """Store a node in memory.

        Args:
            node: The Node object to store.
        """
        self._store[node.key] = node
        logging.debug(f'Wrote node {node.id} to in-memory backend')

    def read_node(self, key: Union[str, bytes]) -> Optional[Node]:
        """Retrieve a node from memory.

        Args:
            key: Node ID as string or bytes.

        Returns:
            Node: The node if found, None otherwise.
        """
        if isinstance(key, str):
            key = key.encode('utf-8')

        node = self._store.get(key)
        if node:
            logging.debug(f'Read node {node.id} from in-memory backend')
        return node

    def get_percent_full(self) -> float:
        """Return percentage of backend utilization.

        Returns:
            float: Always 0.0 for in-memory backend (no capacity constraints).
        """
        return 0.0

    def stat(self) -> dict:
        """Return statistics about the backend.

        Returns:
            dict: Dictionary with 'entries' key containing node count.
        """
        return {'entries': len(self._store)}
