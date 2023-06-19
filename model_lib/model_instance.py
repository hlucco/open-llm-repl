from abc import abstractmethod

class ModelInstance:

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int):
        pass

    @abstractmethod
    def chat(self, max_tokens: int):
        pass
