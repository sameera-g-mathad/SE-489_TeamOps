## Scientific Text Classification using LLMs

## Team Ops

This is a collaborative project to build a scientific text classification model using LLMs. The project is divided into different phases, each focusing on a specific aspect of the model development. The project is hosted on GitHub and the code is available for public use.

- Authors:
  - [@sameera-g-mathad] Sameer Gururaj Mathad (smathad@depaul.edu)
  - [@Mavudiya] Mahendra AVUDIYAPPAN (mavudiya@depaul.edu)

## Project Overview

The goal of this project is to build a scientific text classification model using LLMs. We intend to use the [“knowledgator/Scientific-text-classification"](https://huggingface.co/datasets/knowledgator/Scientific-text-classification) dataset hosted on Hugging-Face. This dataset contains two columns [‘text’, ‘label’], that are self-explanatory. There are 78,631 instances, out of which we plan to use 50,000 subsets. There are 28,641 unique class labels in this dataset. Our focus however is on [statistics, quantum physics, physics, mathematics, high energy physics theory, high energy physics phenomenology, electrical engineering and systems science, condensed matter, computer science, astrophysics] class labels as their distribution and frequency is higher than 4000, and the remaining class labels appear only once in the dataset which is not ideal for training.

## Setup Instructions

Start by cloning the repository:

```bash
git clone https://github.com/sameera-g-mathad/SE-489_TeamOps.git

cd team_ops
```

Then, create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install the dependencies:

## To install the dependencies, this is needed to run the project.

```bash
make install
```

Select the testing_model.ipybook and run the cells to train the model. The model will be saved in the models directory.

To update the dependencies in requirements.txt, run:

```bash
make pipreqs
```

To check for any errors in the code, run:

```bash
make ruff
```

The code we used to build the model is in /notebooks directory. We have provided the yaml files to run the model. The config files are used to set the parameters for the model, such as learning rate, batch size, number of epochs, etc. The config files are used in the training and prediction scripts. Plase change the directory paths in the config files to match your directory structure. The config files are used in the training and prediction scripts. This file will be updated as we progress in the project.

## Data

We have created s3 bucket to store data and models. The keys are not available to everyone. It is the user responsibility to create the s3 bucket and add the keys in .env file.

- To push the data to s3 bucket, run the following command:

```bash
dotenv run -- dvc push -u data-remote
```

- To pull the data from s3 bucket, run the following command:

```bash
dotenv run -- dvc pull -u data-remote
```

- To push the model to s3 bucket, run the following command:

```bash
dotenv run -- dvc push -u model-remote
```

- To pull the model from s3 bucket, run the following command:

```bash
dotenv run -- dvc pull -u model-remote
```

## Contributions

- [@sameera-g-mathad] Sameer Gururaj Mathad - Responsible for the overall project management and coordination from creation to updating readme files. Trained the model and created the config files. Also created s3 bucket to store the data and models and configured the dvc to push the data and models to s3 bucket.

- [@Mavudiya] Mahendra AVUDIYAPPAN - Responsible for processing the data and creating a trainable processed dataset.

## References

Please refer the requirements.txt and requirements_manual.txt files for the list of dependencies used in the project. The requirements.txt file is used to install the dependencies using pip. The requirements_manual.txt file is used to install the dependencies manually. These are the dependencies that are required by the main libraries used in the project. Transformers is the main library that has support for fine-tuning the model provided by Hugging Face.

- Third party libraries used in the project:

  - [transformers](https://huggingface.co/docs/transformers/index)
  - [datasets](https://huggingface.co/docs/datasets/index)
  - [dvc](https://dvc.org/doc/start)
  - [dotenv](https://pypi.org/project/python-dotenv/)
  - [hydra](https://hydra.cc/docs/intro/)
  - [pytorch](https://pytorch.org/get-started/locally/)
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [numpy](https://numpy.org/)
  - [pandas](https://pandas.pydata.org/)

- Dataset
  - [“knowledgator/Scientific-text-classification"](https://huggingface.co/datasets/knowledgator/Scientific-text-classification)
