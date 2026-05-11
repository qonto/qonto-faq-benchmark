# Query the index to obtain the most relevant documents for a given question.
#
# python retriever/<approach>/query.py --index [index.jsonl] [query] >result.json
#
# Returns a list of relevant documents as JSON through stdout.
# Format: list of documents `[{id, score, snippet}]`.

import os
import sys
import json
import argparse
import time
from typing import Any
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

from dataset.jsonl import parse_jsonl


mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def query(query: str, index_path: str, n_docs: int = 2) -> list[dict[str, Any]]:
    """Takes a query and an index file path.
    Returns a list of relevant documents in the form {id: str, score: float}."""
    indexes = parse_jsonl(index_path)
    return top_k(query, indexes, n_docs)


def top_k(query: str, indexes: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Returns {id, score} for the top k most relevant documents."""
    scored_indexes = score_indexes(query, indexes)
    sorted_indexes = sorted(scored_indexes, key=lambda x: x["score"], reverse=True)
    return sorted_indexes[:k]


def score_indexes(query: str, indexes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compares the query to each document in the index.
    Returns a list of {id: int, score: float}."""
    query_embed = embedding([query])[0]
    return [{
        "id": index["id"],
        "score": dot_product(query_embed, index["vector"])
    } for index in indexes]


def embedding(texts: list[str]) -> list[list[float]]:
    """Takes a list of texts to index.
    Return a list of embeddings, one for each text."""
    try:
        resp = mistral.embeddings.create(model="mistral-embed", inputs=texts)
    except SDKError as e:
        if "Status 429" in str(e):
            print(f"Rate limited. Retrying...", file=sys.stderr)
            time.sleep(1)
            return embedding(texts)
        if "Status 500" in str(e):
            print(f"500 error. Retrying...", file=sys.stderr)
            time.sleep(1)
            return embedding(texts)
        else:
            raise e
    return [data.embedding for data in resp.data]


def dot_product(a: list[float], b: list[float]) -> float:
    return sum([a[i] * b[i] for i in range(len(a))])


if __name__ == "__main__":
    # Parse the CLI flags.
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="The index file to help find relevant documents.")
    parser.add_argument("query", nargs="+", help="The query to search for.")
    args = parser.parse_args()

    relevant_docs = query(" ".join(args.query), args.index)
    for doc in relevant_docs:
        print(json.dumps([{"id": doc["id"], "score": doc["score"]}]))
