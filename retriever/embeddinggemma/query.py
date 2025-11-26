# Query the index to obtain the most relevant documents for a given question.
#
# python retriever/embeddinggemma/query.py --index [index.jsonl] [query] >result.json
#
# Returns a list of relevant documents as JSON through stdout.
# Format: list of documents `[{id, score}]`.

import argparse
import json
from typing import Any

from sentence_transformers import SentenceTransformer

from dataset.jsonl import parse_jsonl

MODEL_ID = "google/embeddinggemma-300m"
_model = SentenceTransformer(MODEL_ID)


def query(question: str, index_path: str, n_docs: int = 2) -> list[dict[str, Any]]:
    """Return a list of relevant documents for the given query."""
    indexes = parse_jsonl(index_path)
    return top_k(question, indexes, n_docs)


def top_k(question: str, indexes: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Returns {id, score} for the top k most relevant documents."""
    scores = score_indexes(question, indexes)
    return sorted(scores, key=lambda item: item["score"], reverse=True)[:k]


def score_indexes(question: str, indexes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compares the query to each document in the index.
    Returns a list of {id: int, score: float}."""
    query_embed = embed_query(question)
    return [{
        "id": index["id"],
        "score": dot_product(query_embed, index["vector"])
    } for index in indexes]


def embed_query(question: str) -> list[float]:
    """Returns the embedding for a single query string."""
    embedding = _model.encode_query(
        question,
        normalize_embeddings=True,
    )
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return [float(value) for value in embedding]


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="The index file for document vectors.")
    parser.add_argument("query", nargs="+", help="The query to search for.")
    args = parser.parse_args()

    relevant_docs = query(" ".join(args.query), args.index)
    for doc in relevant_docs:
        print(json.dumps([{"id": doc["id"], "score": doc["score"]}]))
