import torch
import time
import gpt4all
import gc

from model_lib.model_instance import ModelInstance

class Model(ModelInstance):

    def __init__(self) -> None:
        self.model_name = "ggml-gpt4all-j"
        self.model_path = 'ggml-gpt4all-j-v1.3-groovy.bin'
        print("Loading {model_name}...".format(model_name=self.model_name))
        self.model = gpt4all.GPT4All(self.model_path)


    def generate(self, prompt: str, max_tokens: int) -> str:

        messages = [{"role" : "user", "content" : prompt}]
        gen_start = time.time()

        response = self.model.chat_completion(messages, verbose=True)

        gen_time = time.time() - gen_start
        print("Generation Time: " + str(gen_time))

        return response['choices'][0]["message"]["content"]

    def chat(self, max_tokens: int):
        print("Chat with {model_name}".format(model_name=self.model_name))

        while True:
            user_input = input("> ")

            if user_input == "exit":
                exit()
            elif user_input == "swap":
                break

            response = self.generate(user_input, max_tokens)
            print(response)

        # Clearing GPU memory to reset for next model
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        return
