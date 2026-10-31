
import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from nltk.tokenize import wordpunct_tokenize


# ==================================
# Configuration
# ==================================

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "faqs.json"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

FALLBACK = (
    "Sorry, I couldn't find a reliable answer. "
    "Please rephrase your question."
)


# ==================================
# Text Preprocessing
# ==================================

def preprocess(text):
    tokens = wordpunct_tokenize(" ".join(text.split()))

    return " ".join(
        token
        for token in tokens
        if any(char.isalnum() for char in token)
    )


def normalize_vectors(vectors):
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors / np.maximum(norms, 1e-12)


# ==================================
# Semantic FAQ Chatbot
# ==================================

class FAQChatbot:

    def __init__(self, data_path=DATA_PATH, threshold=0.70):
        self.threshold = threshold

        # Load FAQ dataset
        self.faqs = json.loads(
            Path(data_path).read_text(encoding="utf-8")
        )

        self.questions = []
        self.answer_indices = []

        # Collect questions and alternative phrasings
        for index, faq in enumerate(self.faqs):
            examples = [
                faq["question"],
                *faq.get("aliases", [])
            ]

            for question in examples:
                self.questions.append(
                    preprocess(question)
                )
                self.answer_indices.append(index)

        # Load pretrained semantic embedding model
        self.model = TextEmbedding(
            model_name=MODEL_NAME
        )

        # Encode FAQ examples once
        vectors = np.asarray(
            list(self.model.embed(self.questions))
        )

        self.vectors = normalize_vectors(vectors)

        self.last_score = 0.0

    # ==================================
    # Answer Retrieval
    # ==================================

    def reply(self, question):

        # Reset previous similarity score
        self.last_score = 0.0

        # Validate input
        if not isinstance(question, str) or not question.strip():
            return FALLBACK

        # Preprocess user question
        query = preprocess(question)

        if not query:
            return FALLBACK

        # Generate semantic embedding
        query_vector = np.asarray(
            list(self.model.embed([query]))
        )

        query_vector = normalize_vectors(
            query_vector
        )

        # Cosine similarity
        scores = self.vectors @ query_vector[0]

        # Highest score for each FAQ
        faq_scores = np.full(
            len(self.faqs),
            -1.0
        )

        for i, score in enumerate(scores):
            faq_id = self.answer_indices[i]

            faq_scores[faq_id] = max(
                faq_scores[faq_id],
                float(score)
            )

        # Select best FAQ
        best = int(np.argmax(faq_scores))

        best_score = float(faq_scores[best])

        self.last_score = best_score

        # Reject low-confidence matches
        if best_score < self.threshold:
            return FALLBACK

        # Return best matching answer
        return self.faqs[best]["answer"]


# ==================================
# Command-Line Interface
# ==================================

if __name__ == "__main__":

    bot = FAQChatbot()

    print("Semantic FAQ Chatbot")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = bot.reply(question)

        print("Bot:", answer)
        print(f"Similarity: {bot.last_score:.3f}")
