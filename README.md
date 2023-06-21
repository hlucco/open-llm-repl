# Open Source LLM REPL

Small repo containing a REPL for playing with the open source LLMs using torch and 🤗 transformers.

## Installation

1. [Install torch](https://pytorch.org/get-started/locally/) for your operating system (CUDA 11.7 if windows, Default if mac) 

2. `pip install -r requirements.txt`

3. `python main.py`

## Usage

Commands are entered using the `/` prefix. All other inputs are passed into the currently loaded model as a prompt.

## Commands


- `/exit` terminates process
- `/tokens [num_tokens]` update the max allowed tokens for generation
- `/model [model_name]` load specified model
- `/meta [filename]` load a meta prompt template from `/meta`
- `/load [filename]` load and run a predefined set of input from `/load`
- `/list` list all supported model names
