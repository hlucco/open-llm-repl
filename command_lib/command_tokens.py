from command_lib.command import Command
from util.colors import red

class CommandTokens(Command):

    def __init__(self) -> None:
        super().__init__()

    def run(self, repl, args: list[str]) -> str:

        if len(args) != 1:
            return red("ERROR: invalid number of args")

        new_tokens = int(args[0])

        repl.set_tokens(new_tokens)

        return f"Max tokens set to {new_tokens}"

