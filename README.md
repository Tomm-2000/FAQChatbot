# Semantic FAQ Chatbot

A CPU-based Python command-line chatbot that returns answers from a curated FAQ. It embeds the user's question, finds the closest known phrasing, and returns a fallback when the match score is too low.

Built for the CodeAlpha Artificial Intelligence Internship.

## How it works

1. Normalize and tokenize text with NLTK.
2. Embed FAQ questions and aliases using `BAAI/bge-small-en-v1.5` through FastEmbed and ONNX Runtime.
3. Compare the normalized question vector with the FAQ vectors using cosine similarity.
4. Return the matching entry's stored answer when the score reaches the default threshold of `0.70`.

## Setup and run

From the repository root, use a Python 3.12 virtual environment.

```bash
python -m pip install -r requirements.txt
python -m src.chatbot
```

The embedding model may download on first use. Ask a question about the CodeAlpha internship or AI tasks, then try an unrelated question to see the fallback. Type `exit` to close the chat.

## Customize the FAQ

Edit [data/faqs.json](data/faqs.json) to add questions, alternate phrasings, and vetted answers. Restart the CLI to rebuild the in-memory representations. Review matches on known paraphrases and out-of-domain questions before changing the similarity threshold.

## Tests and scope

```bash
python -m unittest discover -s tests -v
```

The suite covers known questions, paraphrases, unsupported topics, and invalid input. The development run recorded four test methods covering 21 initial validation cases. The default similarity threshold is provisional and has not been calibrated on a large independent evaluation set.

| Path | Purpose |
| --- | --- |
| [src/chatbot.py](src/chatbot.py) | Preprocessing, embedding, similarity scoring, and CLI |
| [data/faqs.json](data/faqs.json) | FAQ questions, aliases, and approved answers |
| [tests/test_chatbot.py](tests/test_chatbot.py) | Matching and fallback tests |
