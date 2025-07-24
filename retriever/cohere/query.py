
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
from typing import Any
import cohere

from dataset.jsonl import parse_jsonl

co = cohere.Client(os.environ.get("COHERE_API_KEY"))

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
        response = co.embed(
            texts=texts,
            model="embed-v4.0",
            input_type="search_query"
        )
    except cohere.errors.too_many_requests_error.TooManyRequestsError as e:
        print("Rate limit hit, waiting for 60 seconds before retrying...", file=sys.stderr)
        time.sleep(60)
        return embedding(texts)
    except cohere.core.api_error.ApiError as e:
        print(f"Error during embedding: {e}, retrying after 30 seconds...", file=sys.stderr)
        time.sleep(30)
        return embedding(texts)
    return response.embeddings


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
