from abc import ABC, abstractmethod
from typing import Union, Type, Optional
from jfdb.nodes.node import Node
from pydantic import BaseModel

# TODO: implement a fully in-memory backend
# TODO: implement a LevelDB backend to see if write speed is increased
# TODO: implement backend migration capability

class Backend(BaseModel, ABC):
  name:Optional[str]="default_db"
  node_type:Type[Node]=Node

  @abstractmethod
  def init_backend(self) -> None:
    """Initialize the backend storage system."""
    pass

  @abstractmethod
  def drop_backend(self) -> None:
    """Drop/clear all data from the backend."""
    pass

  @abstractmethod
  def write_node(self, node: Node) -> None:
    """Persist a node to the backend.

    Args:
        node: The node to write.
    """
    pass

  @abstractmethod
  def read_node(self, id: Union[str, bytes]) -> Node:
    """Retrieve a node from the backend.

    Args:
        id: The node identifier (string or bytes).

    Returns:
        The retrieved Node, or None if not found.
    """
    pass

  @abstractmethod
  def get_percent_full(self) -> float:
    """Get storage utilization percentage.

    Returns:
        Percentage of allocated storage used (0-1).
    """
    pass