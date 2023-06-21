from command_lib.command import Command
from util.colors import red

class CommandSetVar(Command):

    def __init__(self) -> None:
        super().__init__()

    def run(self, repl, args: list[str]) -> str:

        if len(args) != 2:
            return red("ERROR: invalid number of args")

        key = args[0]
        val = args[1]

        repl.set_var(key, val)

        return f"Variable {key} set as {val}"
