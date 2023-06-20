from abc import abstractmethod
import gc
import torch
import os

META_DIR = "./meta"

class colors:
    RED = '\033[91m'
    RESET = '\033[0m'

class ModelInstance:

    model_name = ""
    model_instance = None
    tokenizer = None
    model = None

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int):
        pass

    def __parse_input(self, user_input: str) -> str:
        input_tokens = user_input.split(" ")

        for i, token in enumerate(input_tokens):
            if token[0:5] == "<meta":
                meta_name = token.split(":")[1][0:-1]
                found_match = False
                
                for file_name in os.listdir(META_DIR):
                    prompt_name = file_name.split(".")[0]
                    if prompt_name == meta_name:
                        found_match = True

                        with open(os.path.join(META_DIR, file_name), "r") as prompt_file:
                            file_contents = prompt_file.read()
                            input_tokens[i] = file_contents[0:-1]

                if not found_match:
                    print(f"{colors.RED}ERROR: Meta prompt matching {meta_name} was not found{colors.RESET}")
                    del input_tokens[i]

        composed_input = " ".join(input_tokens)
        return composed_input

    def chat(self, max_tokens: int):
        print("Chat with {model_name}".format(model_name=self.model_name))

        while True:
            user_input = input("> ")

            if user_input == "exit":
                exit()
            elif user_input == "tokens":
                max_tokens = int(input("new max token amount: "))
                continue
            elif user_input == "swap":
                break

            composed_input = self.__parse_input(user_input)

            response = self.generate(composed_input, max_tokens)
            print(response)

        # Clearing GPU memory to reset for next model
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return
