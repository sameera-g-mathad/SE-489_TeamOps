import cProfile
import pstats
import io
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
