# Open Source LLM REPL

Small repo containing a REPL for playing with the open source LLMs using torch and 🤗 transformers.

## Installation

1. [Install torch](https://pytorch.org/get-started/locally/) for your operating system (CUDA 11.7 if windows, Default if mac) 

2. `pip install -r requirements.txt`

3. `python main.py`

## Usage

When prompted, enter a string to send it to the selected LLM as a prompt. To leave the REPL, type "exit". Additional commands:
- `swap` change the active model
- `tokens` update the max allowed tokens for generation
