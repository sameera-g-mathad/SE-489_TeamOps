from team_ops import Model


def main():
    """
    Main function to train the model.
    """
    # Initialize the model
    model = Model()

    # Get the dataset
    model.make_data()

    # Train the model
    model.train()


if __name__ == "__main__":
    main()
