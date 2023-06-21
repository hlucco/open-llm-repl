from abc import abstractmethod

class Command:

    # TODO
    # This is hard coded need to update it to be dynamic
    def __init__(self) -> None:
        self.__model_names = [
            "open_llama",
            "gptj",
            "gpt4all",
            "falcon7b-instruct"
        ]

    @abstractmethod
    def run(self, repl, args: list[str]) -> str:
        response = ""
        for name in self.__model_names:
            response += f"{name}\n"
        
        return response
