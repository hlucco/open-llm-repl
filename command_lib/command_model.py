from command_lib.command import Command
from typing import Dict, Type
from model_lib import open_llama, gptj, gptfourall, falcon
from model_lib.model_instance import ModelInstance
from util.colors import red
import gc
import torch

model_names: Dict[str, Type[ModelInstance]] = {
    "open_llama" : open_llama.Model,
    "gptj" : gptj.Model,
    "gpt4all" : gptfourall.Model,
    "falcon7b-instruct": falcon.Model
}

class CommandModel(Command):

    def __init__(self) -> None:
        super().__init__()

    def run(self, repl, args: list[str]) -> str:
        if len(args) != 1:
            return red("ERROR: invalid number of args")

        old_model = repl.get_active_model()
        if old_model:
            del old_model.model, old_model.tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        new_model = model_names[args[0]]()
        repl.set_active_model(new_model)
        response = f"Loaded {new_model.model_name}"
        return response

