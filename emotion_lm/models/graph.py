"""
Module related to graph functionality.
"""
class Graph:
    """
    Implementation for a directed graph data structure.

    nodes: dict[str, any] ex. {"A": <data for node A>, "B": <data for node B>, "C": <data for node C>}
    adjacency_list: dict[str, dict[str, any]] ex. {"A": {"B":  <data for edge AB>, "C": <data for edge AC>}}
    """
    def __init__(self):
        self.nodes = dict()
        self.adjacency_list = dict()

    def clear(self):
        self.nodes = dict()
        self.adjacency_list = dict()
    
    def node_count(self):
        return len(self.nodes)
    
    def edge_count(self):
        return len(self.adjacency_list)
    
    def get_nodes(self):
        return self.nodes
    
    def get_adjacency_list(self):
        return self.adjacency_list
    
    def get_node(self, id):
        return self.nodes[id]
    
    def add_node(self, id, data=None):
        self.nodes[id] = data
    
    def get_edges_for_node(self, start_id):
        return self.adjacency_list[start_id]
    
    def get_edge(self, start_id, end_id):
        return self.adjacency_list[start_id][end_id]

    def add_edge(self, start_id, end_id, metadata = None):
        if start_id not in self.nodes:
            raise KeyError(f"Could not find node with id: '{start_id}'")
        if end_id not in self.nodes:
            raise KeyError(f"Could not find node with id: '{end_id}'")
        if start_id not in self.adjacency_list:
            self.adjacency_list[start_id] = dict()

        self.adjacency_list[start_id][end_id] = metadata
    
    def remove_node(self, id):
        if id not in self.nodes:
            return
        
        del self.nodes[id]
    
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
