from transformers import AutoTokenizer, GPTJForCausalLM
import torch
import time

from model_lib.model_instance import ModelInstance

class Model(ModelInstance):

    def __init__(self) -> None:
        self.model_name = "GPTJ"
        self.model_path = 'EleutherAI/gpt-j-6B'

        print("Loading {model_name}...".format(model_name=self.model_name))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = GPTJForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map='auto',
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device found: {device}".format(device=device))
        if (torch.cuda.is_available()):
            print("GPU: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))

    def generate(self, prompt: str, max_tokens: int) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        inputs = inputs.to('cuda')
        response = ""

        if (type(self.model) == GPTJForCausalLM):
            gen_start = time.time()
            generate_ids = self.model.generate(
                inputs, 
                do_sample=True, 
                temperature=0.9,
                max_length=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id

            )
            gen_time = time.time() - gen_start
            print("Generation Time: " + str(gen_time))

            response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response
