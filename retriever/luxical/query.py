# Query the index to obtain the most relevant documents for a given question.
#
# python retriever/<approach>/query.py --index [index.jsonl] [query] >result.json
#
# Returns a list of relevant documents as JSON through stdout.
# Format: list of documents `[{id, score, snippet}]`.

import argparse
import json
import numpy as np
from typing import Any
from pathlib import Path
import sys

# Add the project root to Python path to import dataset module
sys.path.append(str(Path(__file__).parent.parent.parent))

import luxical.embedder
from dataset.jsonl import parse_jsonl

def get_model_path() -> str:
    """Download and return the path to the luxical model."""
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id='datologyai/luxical-one',
            filename='luxical_one_rc4.npz'
        )
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        # Fallback: try to use a local path if available
        return "luxical_one_rc4.npz"

def query(query: str, index_path: str, n_docs: int = 2) -> list[dict[str, Any]]:
    """Takes a query and an index file path.
    Returns a list of relevant documents in the form {id: str, score: float}."""
    # Load the luxical embedder
    model_path = get_model_path()
    embedder = luxical.embedder.Embedder.load(model_path)

    # Parse the index
    indexes = parse_jsonl(index_path)

    return top_k(query, indexes, n_docs, embedder)

def top_k(query: str, indexes: list[dict[str, Any]], k: int, embedder: luxical.embedder.Embedder) -> list[dict[str, Any]]:
    """Returns {id, score} for the top k most relevant documents."""
    scored_indexes = score_indexes(query, indexes, embedder)
    sorted_indexes = sorted(scored_indexes, key=lambda x: x["score"], reverse=True)
    return sorted_indexes[:k]

def score_indexes(query: str, indexes: list[dict[str, Any]], embedder: luxical.embedder.Embedder) -> list[dict[str, Any]]:
    """Compares the query to each document in the index.
    Returns a list of {id: str, score: float}."""
    # Embed the query
    query_embedding = embedder([query], progress_bars=False)[0]

    # Score each document
    results = []
    for index in indexes:
        doc_embedding = np.array(index["vector"])
        score = cosine_similarity(query_embedding, doc_embedding)
        results.append({
            "id": index["id"],
            "score": score
        })

    return results

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    # Parse the CLI flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="The index file to help find relevant documents.")
    parser.add_argument("query", nargs="+", help="The query to search for.")
    args = parser.parse_args()

    relevant_docs = query(" ".join(args.query), args.index)
    for doc in relevant_docs:
        print(json.dumps([{"id": doc["id"], "score": doc["score"]}]))
