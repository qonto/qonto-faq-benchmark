# Query the index to obtain the most relevant documents for a given question.
#
# python retriever/<approach>/query.py --index [index.jsonl] [query] >result.json
#
# Returns a list of relevant documents as JSON through stdout.
# Format: list of documents `[{id, score, snippet}]`.

import argparse
import bm25s
import json
from typing import Any

def query(query: str, index: bm25s.BM25, n_docs: int = 2) -> list[dict[str, Any]]:
    tokens = bm25s.tokenize(query)
    batch_docs, batch_scores = index.retrieve(tokens, k=n_docs)
    # batch_scores have the form [[score]] of shape (queries, n_docs).
    # batch_docs have the form [[doc_id]] of shape (queries, n_docs).
    docs, scores = batch_docs[0], batch_scores[0]
    return [{"id": id_from_int(docs[i]), "score": scores[i]} for i in range(len(docs))]

def id_from_int(doc_index: int) -> str:
    """Converts a document ID from integer index to ID string format."""
    return f"{doc_index + 1:04d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="The index file to help find relevant documents.")
    parser.add_argument("query", nargs="+", help="The query to search for.")
    args = parser.parse_args()

    relevant_docs = query(" ".join(args.query), args.index)
    for doc in relevant_docs:
        print(json.dumps([{"id": doc["id"], "score": doc["score"]}]))
