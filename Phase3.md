# team_ops

## 1. Continuous Integration & Testing

### Unit Testing with pytest

This phase focuses on setting up continuous integration and testing for the project. We have used python's pytest framework for testing the code. The tests are present in the `tests/` directory. To run the tests, you can use the following command:

#### To run tests regarding data:

This will run the tests related to data validation and processing. We have added tests to check if the directories and files are present, if the data is in the correct format, and if the data is processed correctly.

#### Find the attached image for the test results:

![test_data_image_placeholder](readme_images/test_data.png)

```bash
make test_data
```

#### To run tests regarding training:

This will run the tests related to model training. We have added tests to check if the model is trained, if there exists a save path for the model, if the file has a train method and few other tests and also if the model does not throw any errors during training.

#### Find the attached image for the test results:

![test_train1_image_placeholder](readme_images/test_train_1.png)
![test_train2_image_placeholder](readme_images/test_train_2.png)

```bash
make test_train
```

#### To run tests regarding inference:

We added test cases to run tests on inference of a model. This includes model script, predict method, the return type of the prediction and what the return value is

#### Find the below image for the inference results:

![test_inference1_image_placeholder](readme_images/test_inference1.png)
![test_inference2_image_placeholder](readme_images/test_inference2.png)

```bash
make test_inference
```

### Alternative:

Run all tests at once using make command. This runs all the test specified above in a single go.

#### Find the below image for the results of all test cases:

![test_all1_image_placeholder](readme_images/test_all1.png)
![test_all2_image_placeholder](readme_images/test_all2.png)

```bash
make test_all
```

### GitHub Actions Workflows

We also configured github actions workflows to run all the tests once the pull requests are merged into the main branch. This makes all the tests run in real time ensuring completeness of the code and alerting in case of failure.

We have created a file called `test.yaml` under **.github/worflows** that does all the testing of our code. Point to note, our code needs both model and data to be fetch from external source and needs secrets to be added. This is to access aws s3 bucket and needs to be added into github action secrets.

_The workflow is run on ubuntu, macos and windows for consistency_

#### Example of secrets page.

![github_secrets_example](readme_images/github_secrets_example.png)

### Output of running the workflow( [Link for more details] (https://github.com/sameera-g-mathad/SE-489_TeamOps/actions/runs/15569666753/job/43842281296))

![github_test_workflow_output](readme_images/github_test_workflow_output.png)

### Pre-commit Hooks

We have used python package pre-commit to check and report errors before a commit is made to the github. Our encounters were with trailing spaces in certain files that failed the pre-commit checks which also fixes them. We have added the package in our requirements file so it would be available on install.

#### To create a .pre-commit-config.yaml (This should set up the pre-commit.)

```bash
pre-commit sample-config > .pre-commit-config.yaml
```

#### Example usage:

![pre_commit_example](readme_images/pre_commit_example.png)

## Continuous Docker Building & CML

### Docker Image Automation

We have used github actions to automate the deployment of docker images into dockerhub that makes the images of both train_model and predict_model.
Note: Although we did push the images into dockerhub, the images do not have data or model or any ways or retrieving them, so it will fail. Users are encouraged to download the data, model, and dockerfiles and set an image locally. We realized lately and might fix in the future.

### Figure below shows the workflow example:

![predict_worklow_example](readme_images/predict_worklow_example.png)

### Figure of the images pushed into the dockerhub:

![dockerhub_example](readme_images/dockerhub_example.png)

### Continuous Machine Learning (CML)

Along with having automated deployment of images to dockerhub, we have written `cml.yaml` workflow under **.github/worflows**, that runs the _eval.py_ on main branch. The idea is to train and collect metrics of the model to see the model's performance before deciding to deploy. Although we haven't setup pipeline to train the model due to the requirment of _intense resources_, we did however add a script to evaluate our pretrained model on the test dataset. CML will automatically test the model and publish the results in the comment section. Users are needed to set the `CML_TOKEN` in their secrets that is a **Personal Access Tokens (PATs)** and provide `repo` and `write(discussion)` access for cml to work smoothly.

#### Example image of cml:

![cml_example](readme_images/cml_example.png)

## Deployment on Google Cloud Platform (GCP)

### GCP Artifact Registry

Although we used dockerhub to push images, we still went with GCP Artifact Registry to learn about it. Additionally, we have created a fastAPI server to serve requests from our model.

Firstly, `gcp needs to be installed` on the system. A `cloudbuild.yaml` file is provided on how the images will be built and pushed to gcp. Is is required by the users to _explore gcp and know about creating and linking repositories in cloud build_ before reproducing our work. GCP is provided with the permission to automatically fetch our repository and build an image using cloudbuild.yaml automatically.

#### Summary of image build on gcp

![gcp_build_history](readme_images/gcp_build_history.png)

![gcp_build_summary](readme_images/gcp_build_summary.png)

### Deploying API with FastAPI & GCP Cloud Functions

We have built a fastAPI server under `server/main.py` that serves the model with the incoming requests on `api/predict`. It is a simple
file with a single function.

### Dockerize & Deploy Model with GCP Cloud Run

In gcp cloud run, one can create a service with few options. In our case, we are using the image built in gcp artifact and have set env variables and cpu requirements needed and created the server. As of this writing, our [hosted url](https://server-629521931460.us-central1.run.app) is still active. Is is required by the users to _explore gcp and know about cloud run_ before reproducing our work.

#### Image example of our GCP run service:

![gcp_run_service](readme_images/gcp_run_service.png)

#### Example of inference via POSTMAN:

![postman_inference_example](readme_images/postman_inference_example.png)

### Interactive UI Deployment

We developed an another application only having a streamlit_app.py and requirements.txt that is pushed to a huggingface spaces. The ui is simple with a welcome message, a text area to add the prompt and button to submit the prompt and the returned prediction is displayed on the screen. As of this writing, our frontend app is active and can be accessed through this [link](https://huggingface.co/spaces/MahiJaga/team_ops_frontend).

#### Here is the example of our frontend app in action for different class labels in our dataset:

#### Quantum Physics

![quantum_physics_frontend_example](readme_images/quantum_physics_frontend_example.png)

#### Computer Science

![computer_science_frontend_example](readme_images/computer_science_frontend_example.png)

#### High Energy Physics Theory

![high_energy_physics_theory](readme_images/high_energy_physics_theory.png)

#### Condensed Matter

![condensed_matter](readme_images/condensed_matter.png)

#### Statistics

![statistics_example](readme_images/statistics_example.png)

#### Physics

![physics_example](readme_images/physics_example.png)

#### Astrophysics

![astro_physics_example](readme_images/astro_physics_example.png)

#### High Energy Physics Phenomenology

![high_energy_physics_phenomology](readme_images/high_energy_physics_phenomology.png)

#### Electrical Engineering And Systems Science

![electrical_engineering_and_systems_science](readme_images/electrical_engineering_and_systems_science.png)

#### Mathematics

![mathematics_example](readme_images/mathematics_example.png)
