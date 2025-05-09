# team_ops

## Objective

Large Language Models have become a popular choice for Natural Language Processing, ever since the advent of Decoder-only models. The decoder architecture follows the popular "Transformer” architecture, that was developed by google, that consists of Encoder-Decoder models. Classical examples of such architecture are language translation systems. The encoder encodes the entire sentence of a language into a n-dimensional vector and is passed down to decoder, which in turn uses its input along with the vector of encoder to predict the next word in the target language. In Natural language processing tasks, “context” plays a pivotal role and to embed context into the system, transformer architecture and earlier systems (seq-2-seq with attention) introduced attention mechanisms that calculates the attention weights to attend to different words in the sequence.

Decoder models work on the same concept, but without an encoder. Usually, decoder models published earlier like gpt2 by OpenAI was used to predict next word in the sequence over all the possible words, but they also published decoder models for classification, where the final layer of the model containing embedding layer is replaced by a classification head. This innovation brought a new perspective on text classification, thanks to “self-attention” mechanism.

We plan to use open-source Decoder only models for classification task. Our goal is to successfully train and build a model that uses the Decoder architecture capable of processing raw text or tokenized and accurately predict the final output classifying sequence of words. We intend to utilize publicly hosted models that are available to use for such tasks. This is because Decoder models require pretraining on vast amounts of data available and scraped from the internet. This is done to optimize the weights present in the model. Our classification model would be built (finetuned) on top of a readily available pretrained model, by finetuning the weights to adapt to our task, which is known as ‘Classification Fine-tuning'. (Note: There are Encoder-only models with similar features as well, which we will research and use if needed)

We plan to use “knowledgator/Scientific-text-classification" dataset hosted on Hugging-Face including models as well. This dataset consists of text and labels (academic subjects), the text would otherwise represent. Building a model on this dataset would be useful to classify a text to which category it belongs. An ideal use case would be to tag keywords in papers or articles highlighting the area it represents and would be easier to maintain.

Our expectations are not only learning classification fine-tuning using Decoder models but to gain knowledge on how these models run in production and how they are deployed and monitored, which is the essence of this project.

## Framework

Our primary programming language would be python, since it remains a popular choice for ML related tasks. Since we want to work on LLM’s which require GPU support, we are left with either TensorFlow or Pytorch that are popular for DL tasks and have great GPU support. We will also use Api's from Hugging-Face, that offers great support for NLP tasks both in TensorFlow and Pytorch. Since Pytorch is widely used for NLP tasks, we plan to use it along with Hugging-Face. These are the main libraries needed. Other packages such as NumPy, Pandas, Matplotlib, Scikit-Learn and other packages could be used as needed.

## DataSet

As mentioned above, we intend to use [“knowledgator/Scientific-text-classification"](https://huggingface.co/datasets/knowledgator/Scientific-text-classification) dataset hosted on Hugging-Face. This dataset contains two columns [‘text’, ‘label’], that are self-explanatory. There are 78,631 instances, out of which we plan to use 50,000 subsets. There are 28,641 unique class labels in this dataset. Our focus however is on [statistics, quantum physics, physics, mathematics, high energy physics theory, high energy physics phenomenology, electrical engineering and systems science, condensed matter, computer science, astrophysics] class labels as their distribution and frequency is higher than 4000, and the remaining class labels appear only once in the dataset which is not ideal for training.

## Models

We are using 'distilbert/distilbert-base-uncased' as of now, which has 66 million parameters. Using hugging-face, which swaps the last embedding layer with classification head.

## Installation

Starrt by cloning the repository:

```bash
git clone

cd team_ops
```

Then, create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install the dependencies:

````bash
## To install the dependencies, this is needed to run the project.

```bash
make install
````

Select the testing_model.ipybook and run the cells to train the model. The model will be saved in the models directory.

To update the dependencies in requirements.txt, run:

```bash
make pipreqs
```

To check for any errors in the code, run:

```bash
make ruff
```

The project is configured to run using yaml files, that should be placed in the config directory. The config files are used to set the parameters for the model, such as learning rate, batch size, number of epochs, etc. The config files are used in the training and prediction scripts.

Please change the directory paths in the config files to match your directory structure. The config files are used in the training and prediction scripts. This file will be updated as we progress in the project.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── team_ops  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
