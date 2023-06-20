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

## Meta Prompt Support

The REPL also supports adding the contents of a meta prompt file inline. Any text based file that is added to the `meta` directory can be accessed and added at any part of the prompt. To use a meta prompt, when inputing a prompt into the REPL, add a marker of the following format:

`<meta:[filename]>`

filename should only contain the name of the file, not the name of the file and the extension. For example:

`<meta:txtexample>` and `<meta:example>` 

are valid, while

`<meta:txtexample.txt>` and `<meta:example.md`

will not be found when parsing the prompt. If a meta tag filename is not found while parsing, then a message will be displayed and tag will be removed from the final prompt before it is sent to the model.
