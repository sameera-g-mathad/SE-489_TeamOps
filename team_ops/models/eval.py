from team_ops.models import Model

# This file is to run action by github workflow on push to main branch.

# Initialize the model
model = Model()

# Load the data
model.make_data()

# Load the model
model.load_model()

# Run the model on test data to generate report.
model.eval_cml()
