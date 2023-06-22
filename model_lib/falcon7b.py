from transformers import AutoTokenizer
import transformers
import torch
import time

from model_lib.model_instance import ModelInstance

class Model(ModelInstance):

    def __init__(self) -> None:
        self.model_path = 'tiiuae/falcon-7b'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model_name = "falcon-7b-instruct"

        print("Loading  {model_name}...".format(model_name=self.model_name))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device found: {device}".format(device=device))
        if (torch.cuda.is_available()):
            print("GPU: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))

        self.model = transformers.pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def generate(self, prompt: str, max_tokens: int) -> str:

        response = ""

        gen_start = time.time()
        sequences = self.model(
            prompt,
            max_length=max_tokens,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id = self.tokenizer.eos_token_id
        )

        gen_time = time.time() - gen_start
        print("Generation Time: " + str(gen_time))

        if sequences:
            for seq in sequences:
                if type(seq) == dict:
                    response = seq['generated_text']

        return str(response)
