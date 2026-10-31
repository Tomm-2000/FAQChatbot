# CodeAlpha FAQ Chatbot

An AI-powered FAQ chatbot that retrieves relevant answers using
pretrained semantic embeddings and cosine similarity.

## Features

- Semantic question matching
- Pretrained BGE-small-en-v1.5 embedding model
- FastEmbed with ONNX Runtime (CPU inference)
- NLP tokenization using NLTK
- Multiple question phrasings per FAQ
- Fallback for questions with low similarity
- Interactive command-line interface
- Automated unit tests

## Architecture

User Question
    |
NLP Preprocessing
    |
Semantic Embedding (BGE-small)
    |
Cosine Similarity
    |
Best FAQ Selection
    |
Similarity Threshold
    |
Answer or Fallback

## Tech Stack

- Python 3.12
- FastEmbed
- NumPy
- NLTK
- BAAI/bge-small-en-v1.5

## Installation

Create and activate a Python virtual environment.

Install dependencies:

    python -m pip install -r requirements.txt

The embedding model downloads automatically on first use.

## Run

    python -m src.chatbot

Type `exit` to quit.

## Tests

    python -m unittest discover -s tests -v

Initial validation: 21 passing test cases across 4 test methods.

Tests cover known questions, unknown questions, similar intents,
and invalid inputs.

## Dataset

FAQ questions, aliases, and answers are stored in:

    data/faqs.json

The dataset covers CodeAlpha internship instructions and AI tasks.

## Limitations

- Answers are retrieved from the predefined FAQ dataset.
- The chatbot does not generate new answers.
- The similarity threshold (0.70) is provisional and not
  calibrated on a large independent evaluation dataset.
- Retrieval accuracy may decrease for unsupported topics
  or substantially different question formulations.

## Author

GitHub: Tomm-2000

CodeAlpha Artificial Intelligence Internship Project.
