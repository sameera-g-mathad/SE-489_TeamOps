import cProfile
import pstats
import io
import matplotlib.pyplot as plt
from team_ops import Model


def train():
    """
    Main function to train the model.
    """
    # Initialize the model
    model = Model()

    # Get the dataset
    model.make_data()

    # Train the model
    model.train()

     # Simulate training metrics
    epochs = [1, 2, 3, 4, 5]
    accuracy = [0.6, 0.7, 0.8, 0.85, 0.9]

    # Plot metrics
    plt.plot(epochs, accuracy, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("metrics.png")  # Save the plot as metrics.png


if __name__ == "__main__":
    # Profile inference loop
    pr = cProfile.Profile()
    pr.enable()  # Start profiling

    train()

    pr.disable()  # Stop profiling
    s = io.StringIO()  # Create a string stream to capture the stats
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")  # Sort by cumulative time
    ps.print_stats(25)  # Show top 25 functions by cumulative time
    print(s.getvalue())
