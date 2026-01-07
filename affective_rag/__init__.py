"""Affective-RAG: Memory graph with ALS and Context Path Traversal.

Main entrypoint:
    from affective_rag import MemoryGraph
    
    graph = MemoryGraph()
    graph.add_event("e1", {"text": "...", "timestamp": "..."})
    graph.add_edge("e1", "e2")
    context = graph.retrieve(query, vector_lookup=my_lookup, depth=3)
"""

from .memory_store import MemoryGraph
from .als import calculate_als_score, DEFAULT_ALS_CONFIG, ALSConfig
from .context_path_traversal import execute_context_path_traversal, CPTConfig, ContextPath, CPTResult

__all__ = [
	"MemoryGraph",
	"calculate_als_score",
	"DEFAULT_ALS_CONFIG",
	"ALSConfig",
	"execute_context_path_traversal",
	"CPTConfig",
	"ContextPath",
	"CPTResult",
]

