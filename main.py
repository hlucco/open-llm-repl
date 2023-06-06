import model

MAX_TOKENS = 100

if __name__ == "__main__":

    # init model
    model_instance = model.Model()

    print("Chat with the Open Llama")
    while True:
        user_input = input("> ")

        if user_input == "exit":
            break

        response = model_instance.generate(user_input, MAX_TOKENS)

        print(response)




