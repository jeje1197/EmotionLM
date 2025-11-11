from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def __init__(self, model_name: str):
        """
        Initialize the embedding model based on the string passed in.
        """
        pass

    @abstractmethod
    def generate_embeddings(self, text: str):
        """
        Generate embeddings for the string passed in.
        """
        pass