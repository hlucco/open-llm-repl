from command_lib.command import Command

class CommandList(Command):

    def __init__(self) -> None:
        super().__init__()
        self.__model_names = [
            "open_llama",
            "gptj",
            "gpt4all",
            "falcon7b-instruct"
        ]

    # TODO
    # This is hardcoded for now needs to be made dynamic
    def run(self, repl, args: list[str]) -> str:
        response = ""

        for name in self.__model_names:
            response += f"{name}\n"

        return response
