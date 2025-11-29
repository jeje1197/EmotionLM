from abc import ABC, abstractmethod

class EmbeddingModelWrapper(ABC):
    @abstractmethod
    def call_model():
        raise NotImplementedError("Subclass should implement the call_model method.")

class LanguageModelWrapper(ABC):
    @abstractmethod
    def call_model():
        raise NotImplementedError("Subclass should implement the call_model method.")



