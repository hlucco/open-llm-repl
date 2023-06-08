from model_lib import open_llama, gptj, gptfourall

def run():
    max_tokens = 100

    model_names = {
        "Open Llama" : open_llama, 
        "GPTJ" : gptj,
        "gpt4all" : gptfourall
    }

    print("Choose a model:")
    for i, name in enumerate(model_names):
        print(str((i + 1)) + ". " + name)

    model_selection = int(input("> "))
    selected_pair = (list(model_names.items())[model_selection - 1])
    model_instance = selected_pair[1].Model()

    print("Chat with {model_name}".format(model_name=selected_pair[0]))
    swap = False
    while True:
        user_input = input("> ")

        if user_input == "exit":
            exit()
        elif user_input == "swap":
            swap = True
            break
        elif user_input == "tokens":
            max_tokens = int(input("new max token amount: "))
            continue

        response = model_instance.generate(user_input, max_tokens)
        print(response)

    if swap:
        swap = False
        run()

if __name__ == "__main__":
    run()





