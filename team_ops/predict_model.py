from team_ops import Model


def main():
    """
    Main function to train the model.
    """
    # Initialize the model
    model = Model()

    # Get the dataset
    model.load_model()

    while True:
        prompt = input("Enter the prompt (Type `exit` to quit!!): ")
        if prompt.lower() == "exit":
            break
        prediction = model.predict(prompt)

        print("--------------------")
        print(f"{prediction}")
        print("--------------------")


if __name__ == "__main__":
    main()
