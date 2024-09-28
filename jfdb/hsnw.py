import numpy as np
import logging
import math
from transformers import CLIPProcessor, CLIPModel
import random
import torch
from collections import deque
from pydantic import validate_call, BaseModel
from PIL import Image

from typing import Optional, Type, Iterable, List
from jfdb.backend.backend import Backend
from jfdb.backend.lmdb_backend import LMDBBackend
from jfdb.nodes.node import Node
from jfdb.utils import datastructures

class BFSResult(BaseModel):
	nodes: List[Node]
	priorities: List[float]
	total_traversed_nodes: int

class DataBase():
	# https://www.pinecone.io/learn/series/faiss/hnsw/
	def __init__(
		self,
		L:int,
		M:int,
		ef_construction:int,
		d:int,
		m_L:Optional[float]=None,
		M_max:Optional[int]=None,
		M_max0:Optional[int]=None,
		M_max_tolerance:float=1.0,
		ef:int=1,
		backend:Optional[Backend]=None,
		node_type:Type[Node]=Node
	):
		'''
		Convention is that layer 0 is the base layer

		L: number of layers total
		m_L: level multiplier. Normalization function on top of layer insertion probability function.
		m_L=0 means vectors are only inserted at layer 0. A rule of thumb is to set it at m_L=1/ln(M)
		M: number of connections to make per inserted vertex
		M_max: max number of allowable edges per vertex
		M_max0: max number of allowable edges per vertex in layer 0 only
		M_max_tolerance (float): scale factor. When pruning, we prune to M_max edges but only trigger the prune op if above M_max * M_max_tolerance.
		The idea is to batch prune ops when possible, to speed up insertions
		ef_construction: number of nearest neighbors to return as edge candidates after insertion layer
		d: dimensionality of vectors
		ef: number of nearest neighbors to return before insertion
		'''
		self.L = L
		self.M = M
		self.m_L = m_L or 1/math.log(M)
		self.M_max = M_max or M
		self.M_max0 = M_max0 or M*2
		self.M_max_tolerance = M_max_tolerance
		self.ef_construction = ef_construction
		self.ef = ef
		self.d = d
		self.node_type = node_type

		self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
		self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
		self.backend = backend or LMDBBackend(
			name="default_db",
			preallocated_bytes=int(1e7),
			node_type=self.node_type,
		)
		self.backend.init_backend()

		# self.entry_layer = []
		self.dense_layer = []
		self.layers = {i:[] for i in range(self.L)}  # DEBUGGING


	@property
	def entry_layer(self) -> int:
		return self.L-1
  
	@validate_call
	def bfs_with_max_heap(
		self,
		search_node:Node,
		start_nodes:List[Node],
		ef:int,
		layer:int,
		backend:Backend,
		multi_hop:bool=False
	) -> BFSResult:
		logging.debug(f'Starting bfs on layer {layer} with {len(start_nodes)} start nodes: {[sn.id for sn in start_nodes]}, ef {ef}...')
		_total_traversed_nodes = 0
		max_heap = datastructures.MaxHeap()

		stacked_tensors = torch.stack([i.embedding for i in start_nodes])
		dot_products = torch.sum(search_node.embedding * stacked_tensors, dim=-1).tolist()

		for priority, node in zip(dot_products, start_nodes):
			max_heap.push(priority, node)
		_total_traversed_nodes += 1

		if len(max_heap) >= ef and multi_hop is False:
			logging.debug(f'\tReached ef using search nodes only, returning early')
			while len(max_heap) > ef:
				max_heap.pop()
			objs, prios = max_heap.dump_to_list()
			return BFSResult(nodes=objs, priorities=prios, total_traversed_nodes=_total_traversed_nodes)

		# Queue for BFS
		queue = deque(start_nodes)
		# Set to track visited nodes
		visited = set(start_nodes)

		while queue:
			current_node = queue.popleft()
			candidate_keys = current_node.get_edges(layer)

			# load from backend
			candidates = [backend.read_node(key) for key in candidate_keys]

			# prune any returned neighbors that we've already explored
			candidates = list(set(candidates).difference(visited))  # TODO: should both of these be lists of nodes or keys?
			logging.debug(f'\tGot {len(candidates)} unexplored neighbors from node {current_node.id}: {[c.id for c in candidates]}')

			if len(candidates) > 0:
				stacked_tensors = torch.stack([i.embedding for i in candidates])
				dot_products = torch.sum(search_node.embedding * stacked_tensors, dim=-1).tolist()
				for prio, node in zip(dot_products, candidates):
					_total_traversed_nodes += 1

					if len(max_heap) == ef:
		  				# only add to heap if it is better than the current worst
						if prio > max_heap.peek_priority():
							max_heap.pop()
							max_heap.push(prio, node)
							queue.append(node)
							visited.add(node)
						else:
							# append and search neighbors too
							max_heap.push(prio, node)
							queue.append(node)
							visited.add(node)
		objs, prios = max_heap.dump_to_list()
		logging.debug(f"priorities found by bfs {[round(d, 2) for d in prios]}")

  		# TODO: consider unloading all unused nodes from memory
		return BFSResult(nodes=objs, priorities=prios, total_traversed_nodes=_total_traversed_nodes)

	def insert(self, node:Node, insertion_layer:Optional[int]=None):
		if self.backend.env.stat()['entries'] > 0:
			if self.backend.read_node(node.key) is not None:
				raise IndexError(f'a key of {node.key} already exists in the backend')

		_total_traversed_nodes = 0

		if insertion_layer is None:
			insertion_layer = 0
			if len(self.layers[self.entry_layer]) == 0:
				# this is our first input, always insert into entry layer
				insertion_layer = self.entry_layer
			else:
				# uniform random insertion probability threshold
				insertion_thresh = random.uniform(0, 1)
				for i in reversed(range(self.L)):
					prob = self.probability_function(i)
					if prob > insertion_thresh:
						insertion_layer = i
						break
					else:
						insertion_thresh -= prob  # FAISS does this so so will I
		logging.debug(f'Determined insertion_layer: {insertion_layer}')

		# Graph construction starts at the top layer.
		try:
			candidate_node_keys = [random.choice(self.layers[self.entry_layer])]
			_total_traversed_nodes += 1
		except IndexError:
			# no entry node found - means this is our first insertion
			candidate_node_keys = []

		# Phase 1
		# ----------------
		current_layer = self.entry_layer

		# load from backend. Only have to do this for entry since bfs loads them the rest of the time
		candidate_nodes = [self.backend.read_node(key) for key in candidate_node_keys]

		while current_layer > insertion_layer:
			logging.debug(f'P1: current_layer: {current_layer}')
			# After entering the graph the algorithm greedily traverse across edges,
			#  finding the ef nearest neighbors to our inserted vector q — at this point ef = 1.
			bfs_results = self.bfs_with_max_heap(
				search_node=node,
				start_nodes=candidate_nodes,
				ef=1,
				layer=current_layer,
				backend=self.backend,
			)
			_total_traversed_nodes += bfs_results.total_traversed_nodes - len(candidate_nodes)  # dont double count the candidate nodes
			candidate_nodes = bfs_results.nodes
			current_layer -= 1
			logging.debug(f'P1: updated search node to {candidate_nodes[0].id}')

			# This process is repeated until reaching our chosen insertion layer

		# Phase 2
		# ----------------
		# The ef value is increased to efConstruction (a parameter we set), meaning more nearest neighbors will be returned.
		# In phase two, these nearest neighbors are candidates for the links to the new inserted element q and as entry points to the next layer.

		while current_layer >= 0:
			logging.debug(f'P2: current_layer: {current_layer}')
			self.layers[current_layer].append(node.key)  # debugging
			if len(candidate_nodes) == 0:
				# nothing to seed our search - will happen on first insertion
				logging.debug(f'P2: Got 0 candidate nodes')
				pass
			else:
				bfs_results = self.bfs_with_max_heap(
					search_node=node,
					start_nodes=candidate_nodes,
					ef=self.ef_construction,
					layer=current_layer,
					backend=self.backend,
				)
				_total_traversed_nodes += bfs_results.total_traversed_nodes - len(candidate_nodes)  # dont double count the candidate nodes
				candidate_nodes = bfs_results.nodes
				logging.debug(f'P2: Got {len(candidate_nodes)} candidate nodes')

				# M neighbors are added as links from these candidates — the most straightforward selection criteria are to choose the closest vectors.
				# will also prune extra edges from connections now
				n_keep_edges = self.M_max
				if current_layer == 0:
					n_keep_edges = self.M_max0

				for i, n in enumerate(reversed(candidate_nodes[-self.M:])):
					logging.debug(f'P2 Adding edge to {n.id}')

					# first prune existing node to ensure we can accomodate it
					# NOTE: I'm pruning to M_max and then adding an edge. Will be M_max + 1 edges usually...
					neighbors = [self.backend.read_node(key) for key in n.get_edges(current_layer)]
					if len(neighbors) > n_keep_edges * self.M_max_tolerance:
						sorted_neighbors = n.sort_edges(neighbors=neighbors)
						for neighbor in sorted_neighbors[n_keep_edges:]:
							n.remove_edge(layer=current_layer, node=neighbor)
							self.backend.write_node(neighbor)  # persist change to neighbor edges

					node.add_edge(current_layer, n)  # no need to load since these nodes were necessarily loaded by bfs already
					self.backend.write_node(n)  # but we do need to write them to the backend now that they have edges
			current_layer -= 1

		# persist node
		self.backend.write_node(node)
		logging.info(f'Inserted {node}')
		logging.debug(f'Traversed {_total_traversed_nodes} during insertion')
		return

	@validate_call
	def delete(self, node:Node):
		# TODO: implement
		pass

	def search(self, text:Optional[str]=None, image:Optional[Image.Image]=None) -> Node:
		# TODO: what if we want to return top k results?

		embedding = None
		if text is not None and image is not None:
			raise ValueError('Only one of text, image should be provided for search')

		if text is not None:
			# get embedding from text
			text_input = self.processor(text=[text], return_tensors="pt", padding=True)
			with torch.no_grad():
				embedding = self.model.get_text_features(**text_input)[0]  # take first index since this is batched

		elif image is not None:
			# get embedding from image
			im_input = self.processor(images=image, return_tensors="pt", padding=True)
			with torch.no_grad():
				embeddings = self.model.get_image_features(**im_input)
		else:
			raise ValueError('text and image cannot both be None')
		return self.search_embedding(embedding)

	def search_embedding(self, embedding:Iterable) -> Node:
		# similar to insertion
		_total_traversed_nodes = 0

		dummy_node = Node(
			id='tmp',
			layers=self.L,
			embedding=embedding
		)  # TODO: making a dummy node feels like a design flaw

		current_layer = self.entry_layer
		candidate_node_keys = [random.choice(self.layers[current_layer])]
		_total_traversed_nodes += 1

		# load from backend. Only have to do this for entry since bfs loads them the rest of the time
		candidate_nodes = [self.backend.read_node(key) for key in candidate_node_keys]

		while current_layer >= 0:
			logging.debug(f'current_layer: {current_layer}')
			bfs_results = self.bfs_with_max_heap(
				search_node=dummy_node,
				start_nodes=candidate_nodes,
				ef=1,
				layer=current_layer,
				backend=self.backend,
				multi_hop=True
			)
			_total_traversed_nodes += bfs_results.total_traversed_nodes - len(candidate_nodes)
			candidate_nodes = bfs_results.nodes
			current_layer -= 1

		logging.debug(f'Traversed {_total_traversed_nodes} during search')
		return bfs_results.nodes[0], bfs_results.priorities[0]

	def probability_function(self, layer:int):
		'''
		Probability of insertion at each layer
		'''
		return np.exp(-layer / self.m_L) * (1 - np.exp(-1 / self.m_L))

	def search_brute_force(self, embedding:Iterable):
		# TODO: implement brute force search
		pass
