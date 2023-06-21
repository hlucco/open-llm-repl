from abc import abstractmethod

class Command:

    @abstractmethod
    def run(self, repl, args: list[str]) -> str:
        return "TODO: Command not yet implemented"
