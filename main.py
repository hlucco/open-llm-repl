from typing import Dict, Type
from command_lib.command import Command
from command_lib.command_exit import CommandExit
from command_lib.command_list import CommandList
from command_lib.command_load import CommandLoad
from command_lib.command_meta import CommandMeta
from command_lib.command_model import CommandModel
from command_lib.command_setvar import CommandSetVar
from command_lib.command_tokens import CommandTokens
from model_lib.model_instance import ModelInstance
from util.colors import red

class REPL:

    def __init__(self) -> None:
        self.__max_tokens = 100
        self.__meta_prompt = ""
        self.__commands: Dict[str, Type[Command]] = {
            "model" : CommandModel,
            "exit" : CommandExit,
            "tokens" : CommandTokens,
            "load" : CommandLoad,
            "meta" : CommandMeta,
            "setvar" : CommandSetVar,
            "list" : CommandList
        }
        self.__active_model = None
        self.__env: Dict = {
            "META_DIR" : "./meta",
            "LOAD_DIR" : "./load"
        }

    def set_active_model(self, new_model: ModelInstance):
        self.__active_model = new_model

    def get_active_model(self):
        return self.__active_model

    def set_tokens(self, new_tokens: int):
        self.__max_tokens = new_tokens

    def get_tokens(self):
        return self.__max_tokens

    def set_meta(self, new_meta_contents):
        self.__meta_prompt = new_meta_contents

    def get_meta(self) -> str:
        return self.__meta_prompt

    def set_var(self, key: str, val):
        self.__env[key] = val

    def get_var(self, key: str):
        response = red("ERROR: key not found")

        if key in self.__env:
            response = self.__env[key]

        return response

    def exec(self, user_input: str) -> str:
        response = ""

        if not user_input:
            return red("ERROR: invalid input")

        if user_input[0] == "/":
            user_tokens = user_input.split(" ")
            command_name = user_tokens[0][1:]
            args = user_tokens[1:]

            command_object = self.__commands[command_name]()
            response = command_object.run(self, args)

        elif user_input[0] == "<" and user_input[-1] == ">":
            variable_name = user_input[1:-1]
            response = self.get_var(variable_name)

        else:
            response = red("ERROR: no model loaded, load a model with /model")

            if self.__active_model:
                response = self.__active_model.chat(self, user_input)

        print(response)
        return response

    def run(self):

        while True:
            user_input = input("> ")
            self.exec(user_input)

if __name__ == "__main__":
    REPL().run()
