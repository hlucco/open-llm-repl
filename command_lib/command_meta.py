import os
from command_lib.command import Command
from util.colors import red

class CommandMeta(Command):

    def __init__(self) -> None:
        super().__init__()

    def run(self, repl, args: list[str]) -> str:
        
        if len(args) != 1:
            return red("ERROR: invalid number of args")

        meta_filename = args[0]
        meta_dir = repl.get_var("META_DIR")

        with open(os.path.join(meta_dir, meta_filename), "r") as meta_file:
            file_contents = meta_file.read()
            repl.set_meta(file_contents)

        return f"{meta_filename} loaded as meta prompt"
    
