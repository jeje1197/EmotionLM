import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from emotion_lm.models.graph import Graph

def test_simple_graph():
    graph = Graph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_node("D")

    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("A", "D")

    print(graph.stringify())