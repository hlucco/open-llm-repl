from command_lib.command import Command


class CommandExit(Command):

    def __init__(self) -> None:
        super().__init__()

    def run(self, repl, args: list[str]) -> str:
        exit()

