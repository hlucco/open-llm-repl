from transformers import AutoTokenizer, GPTJForCausalLM
import torch
import time

class Model:

    def __init__(self) -> None:
        print("Loading GPTJ...")
        self.model_path = 'EleutherAI/gpt-j-6B'

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = GPTJForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map='auto',
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device found: {device}".format(device=device))
        if (torch.cuda.is_available()):
            print("GPU: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))

    def generate(self, prompt: str, max_tokens: int) -> str:

        meta_prompt = """### Human:
You are an artificial assistant that gives facts based answers.
You strive to answer concisely.
When you're done responding, add a "Review" section and create and append a terse review to the response.
In your review, you review the response to fact check it and point out any inaccuracies.
Be analytical and critical in your review, and very importantly, don't repeat parts of your answer.

{prompt}

### Assistant:""".format(prompt=prompt)

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
