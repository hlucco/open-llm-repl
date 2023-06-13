from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time

class Model:

    def __init__(self) -> None:
        print("Loading  falcon40b...")
        # self.model_path = 'tiiuae/falcon-40b-instruct'
        self.model_path = 'tiiuae/falcon-7b-instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device found: {device}".format(device=device))
        if (torch.cuda.is_available()):
            print("GPU: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))

        self.pipeline = transformers.pipeline(
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
        sequences = self.pipeline(
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
                print(seq)
                response = seq['generated_text']

        return str(response)
