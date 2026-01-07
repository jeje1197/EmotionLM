"""Demonstration application showing how to integrate affective_rag with ChromaDB.

This example is separate from the core library to keep dependencies clean.
It shows how to build a simple `vector_lookup` adapter that plugs a Chroma
collection into MemoryGraph.retrieve().

Install optional dependency:
    pip install chromadb

Run from project root:
    python -m examples.chroma_demo
"""
