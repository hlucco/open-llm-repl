import os
from command_lib.command import Command
from util.colors import red

class CommandLoad(Command):

    def __init__(self) -> None:
        super().__init__()

    def run(self, repl, args: list[str]) -> str:

        if len(args) != 1:
            return red("ERROR: invalid number of args")

        load_file_name = args[0]
        load_dir = repl.get_var("LOAD_DIR")

        with open(os.path.join(load_dir, load_file_name), "r") as load_file:

            command_list = load_file.read()
            command_lines = command_list.split("\n")

            for line in command_lines:
                repl.exec(line)

        return f"SUCCESS: all commands in {load_file_name} run"
