import torch
import time
import gpt4all

class Model:

    def __init__(self) -> None:
        print("Loading gpt4all...")
        self.model_path = 'ggml-gpt4all-j-v1.3-groovy.bin'
        self.model = gpt4all.GPT4All(self.model_path)

    def generate(self, prompt: str, max_tokens: int) -> str:

        messages = [{"role" : "user", "content" : prompt}]
        response = self.model.chat_completion(messages, verbose=True)

        return response['choices'][0]["message"]["content"]

