"""Directed graph memory graph implementation."""

import networkx as nx
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pydantic import BaseModel

class MemoryEvent(BaseModel):
    id: str
    text: str
    semantic_vec: Optional[List[float]] = None
    emotional_vec: Optional[List[float]] = None
    timestamp: Optional[str] = None

class MemoryGraph:
    """A simple NetworkX-backed directed memory graph.

    Nodes are stored as node attributes (a dict). Each node should include an 'id' key.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._logging_enabled = False
        self._logger_provider = None

    def add_event(self, event_id: str, event_data: Dict[str, Any]) -> None:
        """Add or update an event node.

        event_data is a dictionary (may include semantic_vec, emotional_vec, timestamp, text, etc.).
        """
        data = dict(event_data)
        data.setdefault("id", event_id)
        self.graph.add_node(event_id, **data)

    def add_edge(self, from_event: str, to_event: str, **attrs) -> None:
        """Add a directed edge from `from_event` to `to_event` with optional attributes."""
        self.graph.add_edge(from_event, to_event, **attrs)

    def get_event(self, event_id: str) -> Dict[str, Any]:
        """Return a shallow copy of the stored event attributes."""
        return dict(self.graph.nodes[event_id])

    def neighbors(self, event_id: str) -> List[str]:
        """Return a list of successor node ids."""
        return list(self.graph.successors(event_id))

    def predecessors(self, event_id: str) -> List[str]:
        """Return a list of predecessor node ids."""
        return list(self.graph.predecessors(event_id))

    def nodes(self) -> List[str]:
        return list(self.graph.nodes)

    def edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        return list(self.graph.edges(data=True))

    def has_node(self, event_id: str) -> bool:
        return self.graph.has_node(event_id)

    def subgraph(self, nodes: Iterable[str]) -> nx.DiGraph:
        return self.graph.subgraph(nodes).copy()

    def set_logging(self, enabled: bool, logger_provider: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Configure logging for context path traversal operations.
        
        Args:
            enabled: Whether to enable logging.
            logger_provider: Callable that accepts a dict with log data.
                If None when enabled=True, logs will be printed to console.
        """
        self._logging_enabled = enabled
        self._logger_provider = logger_provider if logger_provider is not None else print

    def retrieve(
        self,
        query: Any,
        *,
        vector_lookup: Callable[[Any], Sequence[Tuple[str, float]]],
        cpt_config: Optional["CPTConfig"] = None,
        event_data_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        """Run Context Path Traversal on this graph.

        Args:
            query: Query object passed to vector_lookup (text, vector, etc.).
            vector_lookup: Callable that returns [(node_id, score), ...] for the query.
            cpt_config: Optional CPTConfig for fine-grained control (max_depth, seed_nodes,
                max_neighbors, min_als_score, cache_event_data, etc.). Defaults to CPTConfig().
            event_data_provider: Optional callable (event_id) -> dict that fetches
                event data from an external source. If None, reads from this graph.
            **kwargs: Forwarded to execute_context_path_traversal (e.g., score_fn).

        Returns:
            CPTResult with context paths ranked by score.
        """
        from .context_path_traversal import (  # local import to avoid cycles
            CPTConfig,
            CPTResult,
            execute_context_path_traversal,
        )

        config = cpt_config if cpt_config is not None else CPTConfig()
        logger = self._logger_provider if self._logging_enabled else None
        
        return execute_context_path_traversal(
            memory_store=self,
            query=query,
            vector_store_lookup_function=vector_lookup,
            cpt_config=config,
            event_data_provider=event_data_provider,
            logger=logger,
            **kwargs,
        )


# Backwards-compat alias
MemoryStore = MemoryGraph

