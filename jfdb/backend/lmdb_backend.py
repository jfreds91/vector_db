from jfdb.backend.backend import Backend
from jfdb.nodes.node import Node
from typing import Union, Any, Optional
import lmdb
import logging

class LMDBBackend(Backend):
  '''
  Note: current node structure with len 512 vectors means each node is 12kb,
  which is bigger than the 4kb page size chosen by LMDB (somehow dictate by
  underlying system pagination - I can't easily change it). That means every
  node is associated with an overflow page, which is not ideal. Increases I/O

  Not the end of the world but we can improve
  '''
  env:Optional[Any]=None  # don't bother supplying this, gets populated during init
  preallocated_bytes:int=int(1e6)

  def init_backend(self):
    # Create an LMDB environment
    self.env = lmdb.open(self.name, map_size=self.preallocated_bytes)
    logging.info(f'Initialized {self.env.path()}')

    # TODO: if backend exists but a user specified a different preallocated bytes, fail loudly

  def drop_backend(self):
    if self.env is None:
      self.init_backend()
      # raise RuntimeError('Cannot drop backend if it has not been initialized. Run init_backend first')

    # Begin a write transaction to clear the database
    with self.env.begin(write=True) as txn:
      txn.drop(db=self.env.open_db(), delete=False)

    logging.warning('Emptied database index')
    # Optionally: Sync and close
    self.env.sync()
    self.env.close()

  def write_node(self, node:Node):
    # Storing the Node object in LMDB using pickle
    with self.env.begin(write=True) as txn:
      txn.put(node.key, node.serialize())  # Store in LMDB
    logging.debug(f'Wrote key {node.key} to {self.env.path()}')

  def read_node(self, key:Union[str, bytes]) -> Node:
    # Retrieving the Node object from LMDB
    with self.env.begin() as txn:
      serialized_node = txn.get(key)  # Retrieve serialized data
      if serialized_node:
        node = self.node_type.deserialize(serialized_node)
        logging.debug(f'Read key {node.key} from {self.env.path()}')
        return node
    return None

  def get_percent_full(self) -> float:
    '''
    return value is between 0 and 1
    '''
    # Get the current map size (allocated size)
    map_size = self.env.info()['map_size']

    # Get the size of the data currently in the database
    used_size = self.env.stat()['psize'] * (self.env.stat()['leaf_pages'] + self.env.stat()['overflow_pages'])

    # Calculate percentage usage
    return used_size / map_size
