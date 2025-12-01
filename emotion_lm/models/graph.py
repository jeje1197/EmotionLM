

class Node:
    def __init__(self, id, memory):
        self.id = id
        self.memory = memory

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

    
