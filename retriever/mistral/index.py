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
from mistralai import Mistral, models

from dataset.documents import DocumentSet

mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def index(docs: DocumentSet) -> None:
    docs_meta = docs.read_metadata()
    batch_size = 4
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
    # Currently, the embedding API only supports up to 8192 bytes per document.
    texts = [text[:8192] for text in texts]
    try:
        resp = mistral.embeddings.create(model="mistral-embed", inputs=texts)
    except models.sdkerror.SDKError as e:
        if "Status 429" in str(e):
            print(f"Rate limited. Retrying...")
            time.sleep(1)
            return embedding(texts)
        else:
            raise e
    return [data.embedding for data in resp.data]


if __name__ == "__main__":
    documents_folder = sys.argv[1]
    docs = DocumentSet(documents_folder)
    index(docs)
