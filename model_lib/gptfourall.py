import time
import gpt4all

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
