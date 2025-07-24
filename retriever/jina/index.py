# Index the dataset/documents and output the index in stdout.
#
# python retriever/<approach>/index.py <documents> >index.jsonl
#
# The documents must be a folder in the same format as dataset/documents.
# Format: for each document, `{id, vector}`.

import sys
import os
import json
import time
from itertools import islice
from tqdm import tqdm
from typing import Any
import requests

from dataset.documents import DocumentSet

def index(docs: DocumentSet) -> None:
    docs_meta = docs.read_metadata()
    batch_size = 32
    # Loop through batch_size lines at a time.
    for batch in tqdm(iter(lambda: list(islice(docs_meta, batch_size)), []), desc=f"Indexing documents in {documents_folder}"):
        index_doc_batch(batch, docs)


def index_doc_batch(docs_batch: list[dict[str, Any]], docs: DocumentSet) -> None:
    # Read the markdown files.
    md_contents = [docs.get_from_meta(m) for m in docs_batch]

    # Index them.
    embed = embedding(md_contents)

    # Output the document's index information.
    for i in range(len(docs_batch)):
        index = {"id": docs_batch[i]["id"], "vector": embed[i]}
        print(json.dumps(index))


# Take a list of texts to index.
# Return a list of embeddings, one for each document.
def embedding(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}"
        },
        data=json.dumps({
            "model": "jina-embeddings-v4",
            "task": "text-matching",
            "input": [{"text": text} for text in texts]
        })
    ).json()
    return [item['embedding'] for item in response['data']]


if __name__ == "__main__":
    documents_folder = sys.argv[1]
    docs = DocumentSet(documents_folder)
    index(docs)
