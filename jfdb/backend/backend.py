from abc import ABC, abstractmethod
from typing import Union, Type, Optional
from jfdb.nodes.node import Node

# TODO: implement a fully in-memory backend
# TODO: implement a LevelDB backend to see if write speed is increased
# TODO: implement backend migration capability

class Backend(ABC):

  def __init__(self, name:Optional[str]="default_db", node_type:Type[Node]=Node):
    self.name = name
    self.node_type = node_type

  @abstractmethod
  def init_backend(self):
    pass

  @abstractmethod
  def drop_backend(self):
    pass

  @abstractmethod
  def write_node(self, node:Node):
    pass

  @abstractmethod
  def read_node(self, id:Union[str, bytes]) -> Node:
    pass

  @abstractmethod
  def get_percent_full(self) -> float:
    pass