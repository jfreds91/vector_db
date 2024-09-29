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

    def add_edge(self, layer:int, node:Node, _recurse:bool=True):
        assert isinstance(node, Node)

        logging.debug(f'adding {node.key} to {self} in layer {layer}')
        if node.key not in self.layer_edges[layer]:
            self.layer_edges[layer].append(node.key)
        else:
            raise KeyError(f"{self.id} already has an edge to {node.key}!")
        if _recurse:
            node.add_edge(layer=layer, node=self, _recurse=False)

    def remove_edge(self, layer:int, node:Node, _recurse:bool=True):
        logging.debug(f'Removing {node.key} from {self} in layer {layer}')

        self.layer_edges[layer].remove(node.key)
        if _recurse:
            node.remove_edge(layer=layer, node=self, _recurse=False)

    def get_edges(self, layer:int) -> Union[bytes, str]:
        keys = self.layer_edges[layer]
        assert all([type(key)==bytes for key in keys])
        return keys


    def sort_edges(self, neighbors:List[Node], method:str='distance') -> List[Node]:
        '''
        Given a start node, return a ranked ordering of edges. This will be used
        to prune edges from nodes which have too many.

        The naive approach is to rank by distance metric.
        TODO: More advanced approaches can:
            - analyze the graph to ensure no orphans
            - return a subset of nodes with max angle between edges
        '''
        assert all([isinstance(neighbor, Node) for neighbor in neighbors])
        # return closest neighbors
        stacked_tensors = torch.stack([i.embedding for i in neighbors])
        dot_products = torch.sum(self.embedding * stacked_tensors, dim=-1).tolist()

        if method=='distance':
            return [node for _priority, node in zip(dot_products, neighbors)]
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