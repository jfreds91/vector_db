from __future__ import annotations  # allows forward reference
from collections import defaultdict
from typing import Optional, List, Dict, Union, Iterable
import pickle
import logging
import torch
import warnings
from dataclasses import dataclass, field

# Ignore FutureWarning: torch.load should have weights_only=True to prevent arbitrary code execution
warnings.simplefilter(action='ignore', category=FutureWarning)

@dataclass
class Node():
    id:str
    layers:int
    embedding:Optional[Iterable]=None
    # embedding:Optional[torch.Tensor]=None
    filepath:Optional[str]=None
    layer_edges:Dict[int, List[Union[str, bytes]]] = field(default_factory=lambda: defaultdict(list))

    @property
    def byte_id(self):
        return self.id.encode('utf-8')

    @property
    def key(self):
        return self.byte_id

    def __repr__(self):
        return self.id

    def add_edge(self, layer: int, node: Node, _recurse: bool = True) -> None:
        """Add an edge to another node in the specified layer.

        Creates a bidirectional edge by default. Use _recurse=False to create
        unidirectional edges (for internal recursive calls).

        Args:
            layer: Layer index at which to add the edge.
            node: Target node to connect to.
            _recurse: If True, adds reciprocal edge on target node. Default True.

        Raises:
            AssertionError: If node is not a Node instance.
        """
        assert isinstance(node, Node)

        logging.debug(f'adding {node.key} to {self} in layer {layer}')
        if node.key not in self.layer_edges[layer]:
            self.layer_edges[layer].append(node.key)
        else:
            logging.warning(f"{self.id} already has an edge to {node.id}. Skipping duplicate edge.")
            return
        if _recurse:
            node.add_edge(layer=layer, node=self, _recurse=False)

    def remove_edge(self, layer: int, node: Node, _recurse: bool = True) -> None:
        """Remove an edge to another node in the specified layer.

        Removes bidirectional edges by default. Logs a warning if edge doesn't exist
        instead of raising an exception.

        Args:
            layer: Layer index from which to remove the edge.
            node: Target node to disconnect from.
            _recurse: If True, removes reciprocal edge on target node. Default True.
        """
        logging.debug(f'Removing {node.key} from {self} in layer {layer}')

        if node.key in self.layer_edges[layer]:
            self.layer_edges[layer].remove(node.key)
        else:
            logging.warning(f'Attempted to remove non-existent edge from {self.id} to {node.id} in layer {layer}')
        if _recurse:
            node.remove_edge(layer=layer, node=self, _recurse=False)

    def get_edges(self, layer: int) -> List[bytes]:
        """Get all edge keys at the specified layer.

        Args:
            layer: Layer index from which to retrieve edges.

        Returns:
            List[bytes]: List of byte-encoded node IDs this node connects to at the layer.

        Raises:
            AssertionError: If any key is not of type bytes.
        """
        keys = self.layer_edges[layer]
        assert all([type(key)==bytes for key in keys])
        return keys


    def sort_edges(self, neighbors: List[Node], method: str = 'distance') -> List[Node]:
        """Return neighbors ranked by similarity to this node.

        Used to prune excessive edges by selecting the closest neighbors.

        Args:
            neighbors: List of neighboring nodes to rank.
            method: Ranking method to use. Currently only 'distance' is supported. Default 'distance'.

        Returns:
            List[Node]: Neighbors sorted by similarity in descending order (most similar first).

        Raises:
            NotImplementedError: If an unsupported ranking method is specified.
            AssertionError: If any neighbor is not a Node instance.

        TODO:
            More advanced approaches could:
            - Analyze the graph to ensure no orphaned nodes
            - Return a subset of nodes with maximum angle between edges
        """
        assert all([isinstance(neighbor, Node) for neighbor in neighbors])
        # return closest neighbors
        stacked_tensors = torch.stack([i.embedding for i in neighbors])
        dot_products = torch.sum(self.embedding * stacked_tensors, dim=-1).tolist()

        if method=='distance':
            return [node for _priority, node in sorted(zip(dot_products, neighbors), key=lambda x: x[0], reverse=True)]
        else:
            raise NotImplementedError(f'method: {method} is not implemented')


    def __lt__(self, other):
        if isinstance(other, Node):
            return self.id < other.id
        raise NotImplementedError(f'Node.__lt__() is undefined for {type(other)}')

    def __eq__(self, other):
        # Define equality based on unique attributes (e.g., id)
        if isinstance(other, Node):
                return self.id == other.id
        return False

    def __hash__(self):
            # Combine the hashes of attributes to ensure a unique and consistent hash
            return hash(self.id)

    def __gt__(self, other):
        logging.warning(f'Someone just tried to compare nodes {self} > {other}')

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, serialized_node:bytes) -> Node:
        obj = pickle.loads(serialized_node)
        assert isinstance(obj, cls)
        return obj