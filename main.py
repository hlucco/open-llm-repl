from model_lib import open_llama, gptj, gptfourall, falcon

def run():
    max_tokens = 100

    model_names = {
        "Open Llama" : open_llama,
        "GPTJ" : gptj,
        "gpt4all" : gptfourall,
        "falcon7b-instruct": falcon
    }

    while True:
        print("Choose a model:")

        for i, name in enumerate(model_names):
            print(str((i + 1)) + ". " + name)

        model_selection = input("> ")
        if model_selection == "tokens":
            max_tokens = int(input("new max token amount: "))
            continue
        elif model_selection == "exit":
            break

        selected_pair = (list(model_names.items())[int(model_selection) - 1])
        model_instance = selected_pair[1].Model()

        model_instance.chat(max_tokens)

if __name__ == "__main__":
    run()
