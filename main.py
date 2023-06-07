import open_llama
import gptj

MAX_TOKENS = 100

if __name__ == "__main__":

    model_names = {
        "Open Llama" : open_llama, 
        "GPTJ" : gptj
    }

    model_name = ""
    print("Choose a model:")
    for i, name in enumerate(model_names):
        print(str((i + 1)) + ". " + name)

    model_selection = int(input("> "))
    selected_pair = (list(model_names.items())[model_selection - 1])
    model_instance = selected_pair[1].Model()

    print("Chat with {model_name}".format(model_name=selected_pair[0]))
    while True:
        user_input = input("> ")

        if user_input == "exit":
            break

        response = model_instance.generate(user_input, MAX_TOKENS)

        print(response)




