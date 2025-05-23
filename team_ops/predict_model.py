import cProfile
import pstats
import io
from prometheus_client import start_http_server, Summary, Counter
from team_ops import Model

# Define a metric to track time spent and requests made
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")
REQUEST_COUNT = Counter("request_count", "Total number of requests")
REQUEST_ERROR = Counter("request_error", "Total number of errors")


@REQUEST_TIME.time()
def inference():
    """
    Main function to train the model.
    """
    # Initialize the model
    model = Model()

    # Get the dataset
    model.load_model()

    while True:
        try:
            prompt = input("Enter the prompt (Type `exit` to quit!!): ")
            if prompt.lower() == "exit":
                break
            prediction = model.predict(prompt)

            print("--------------------")
            print(f"{prediction}")
            print("--------------------")
            REQUEST_COUNT.inc()  # Increment the request count

        except Exception as e:
            REQUEST_ERROR.inc()  # Increment the error count
            raise e


if __name__ == "__main__":
    start_http_server(8000)

    # Profile inference loop
    pr = cProfile.Profile()
    pr.enable()  # Start profiling

    inference()

    pr.disable()  # Stop profiling
    s = io.StringIO()  # Create a string stream to capture the stats
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")  # Sort by cumulative time
    ps.print_stats(25)  # Show top 25 functions by cumulative time
    print(s.getvalue())
