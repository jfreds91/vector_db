from __future__ import annotations  # allows forward reference
from typing import Iterable
from collections import defaultdict
from typing import Optional, List, Dict, Union
import pickle
import logging
import torch
from pydantic import validate_call

class Node():

	def __init__(self, id, layers:int, embedding:Optional[Iterable]=None, filepath:Optional[str]=None,):
		'''
		Layer edges is the key of another Node, not the node object itself
		'''
		self.id=id
		self.embedding = embedding
		self.layers = layers
		self.filepath = filepath
		self.layer_edges:Dict[int, List[Union[str, bytes]]] = defaultdict(list)

	@property
	def byte_id(self):
		return self.id.encode('utf-8')

	@property
	def key(self):
		return self.byte_id

	def __repr__(self):
		return self.id

	@validate_call
	def add_edge(self, layer:int, node:Node, _recurse:bool=True):

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

	@validate_call
	def get_edges(self, layer:int) -> Union[bytes, str]:
		keys = self.layer_edges[layer]
		assert all([type(key)==bytes for key in keys])
		return keys


	@validate_call
	def sort_edges(self, neighbors:List[Node], method:str='distance') -> List[Node]:
		'''
		Given a start node, return a ranked ordering of edges. This will be used
		to prune edges from nodes which have too many.

		The naive approach is to rank by distance metric.
		TODO: More advanced approaches can:
			- analyze the graph to ensure no orphans
			- return a subset of nodes with max angle between edges
		'''
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
	@validate_call
	def deserialize(cls, serialized_node:bytes) -> Node:
		return pickle.loads(serialized_node)