# Index the dataset/documents and output the index in stdout.
#
# python retriever/<approach>/index.py <documents> >index.jsonl
#
# The documents must be a folder in the same format as dataset/documents.
# Format: for each document, `{id, vector}`.

import sys
import tqdm
import bm25s
from typing import Any

from dataset.documents import DocumentSet

def index(docs: DocumentSet, output_path: str) -> bm25s.BM25:
    # Indexer setup.
    indexer = bm25s.BM25()

    # Dataset handling.
    docs_meta = docs.read_metadata()
    md_contents = [docs.get_from_meta(m) for m in docs_meta]
    tokens = bm25s.tokenize(md_contents)
    indexer.index(tokens)
    indexer.save(output_path)
    return indexer

def load_index(index_path: str) -> bm25s.BM25:
    return bm25s.BM25.load(index_path, mmap=True)

if __name__ == "__main__":
    documents_folder = sys.argv[1]
    output_path = sys.argv[2]
    docs = DocumentSet(documents_folder)
    index(docs, output_path)
