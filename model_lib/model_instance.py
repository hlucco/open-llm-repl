from abc import abstractmethod
import re

class ModelInstance:
    
    model_name = ""
    model_instance = None
    tokenizer = None
    model = None

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        pass

    def chat(self, repl, user_input: str) -> str:
        meta_contents = repl.get_meta()

        if not meta_contents:
            return self.generate(user_input, repl.get_tokens())

        meta_vars = list(re.finditer(r"<([^>]+)>", meta_contents))
        while len(meta_vars) != 0:
            match = meta_vars[0]
            key = match.group(1)
            start_idx = match.start()
            end_idx = match.end()

            val = repl.get_var(key)

            if key == "USER_INPUT":
                val = user_input

            meta_contents = meta_contents[:start_idx] + val + meta_contents[end_idx:]
            show_contents = repl.get_var("SHOW_PROMPT")
            if show_contents:
                print(meta_contents)
            meta_vars = list(re.finditer(r"<([^>]+)>", meta_contents))

        response = self.generate(meta_contents, repl.get_tokens())
        return response
