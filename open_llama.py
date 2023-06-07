from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import time

class Model:

    def __init__(self) -> None:
        self.model_path = 'openlm-research/open_llama_7b_700bt_preview'
        # self.model_path = 'openlm-research/open_llama_3b_600bt_preview'

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map='auto',
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device found: {device}".format(device=device))
        if (torch.cuda.is_available()):
            print("GPU: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))

    def generate(self, prompt: str, max_tokens: int) -> str:

        meta_prompt = """
### Instruction: 
The prompt below is a question to answer, a task to complete, or a conversation 
to respond to; decide which and write an appropriate response.
            
### Prompt: 
{prompt}

### Response:
        """.format(prompt=prompt)

        inputs = self.tokenizer(meta_prompt, return_tensors="pt")
        inputs = inputs.to('cuda')
        response = ""

        if (type(self.model) == LlamaForCausalLM):
            gen_start = time.time()
            generate_ids = self.model.generate(inputs.input_ids, max_length=max_tokens)
            gen_time = time.time() - gen_start
            print("Generation Time: " + str(gen_time))

            response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response

    def dep_generate(self, prompt: str, max_tokens: int) -> str:

        response = ""
        tokenize_start = time.time()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        tokenize_time = time.time() - tokenize_start

        print("Tokenization Time: " + str(tokenize_time))

        if (type(self.model) == LlamaForCausalLM):

            gen_start = time.time()
            generation_output = self.model.generate(
                input_ids=input_ids, max_new_tokens=50
            )

            gen_time = time.time() - gen_start
            print("Generation Time: " + str(gen_time))

            print(generation_output)

            output = self.tokenizer.decode(generation_output[0])
            raw_output = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print("Output Len: " + str(len(output.split(" "))))

            # print("=====================")
            # print(output)
            # print("=====================")
            # print(raw_output)
            # print("=====================")
            # print(raw_output[0])
            response = output

        return response



