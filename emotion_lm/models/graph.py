"""
Node: {id: str, data: any}
Edge: {node_start, node_end}

nodes: set
adjacency list: dict[id:str, dict[id: str, data]]
"""
class Graph:
    """
    Implementation for a directed graph data structure.
    """
    def __init__(self):
        self.nodes = set()
        self.adjacency_list = dict()
    
    def node_count(self):
        return len(self.nodes)
    
    def edge_count(self):
        return len(self.adjacency_list)
    
    def get_node(self, id):
        return self.nodes[id]
    
    def add_node(self, id):
        self.nodes.add(id)

    def add_edge(self, start_id, end_id, data = None):
        if start_id not in self.nodes:
            raise ValueError(f"Could not find node with id: '{start_id}'")
        if end_id not in self.nodes:
            raise ValueError(f"Could not find node with id: '{end_id}'")
        if start_id not in self.adjacency_list:
            self.adjacency_list[start_id] = dict()

        self.adjacency_list[start_id][end_id] = data
    
    def remove_node(self, id):
        self.nodes.remove(id)
    
    def remove_edge(self, start_id, end_id):
        if start_id not in self.adjacency_list:
            return
        
        del self.adjacency_list[start_id][end_id]
        if len(self.adjacency_list[start_id]) == 0:
            del self.adjacency_list[start_id]
    
    def stringify(self):
        return f"""\
Graph:
Number of Nodes: {self.node_count()} Number of Edges: {self.edge_count()}
Nodes: {self.nodes}
Edges: {self.adjacency_list}
"""
    
  





class Node:
    def __init__(self, id):
        self.id = id

class Edge:
    def __init__(self, from_node, to_node, type, metadata, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.type = type # caused, etc.
        self.metadata = metadata # causal_strength, temporal_proximity, emotional_intensity weights, confidence
        # weight: optional computed value based on metadata

class MemoryGraph:
    def __init__(self):
        self.nodes = {} # id -> Node
        self.edges = [] # id -> {A, B} 

    
