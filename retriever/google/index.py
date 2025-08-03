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
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from dataset.documents import DocumentSet

google_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def index(docs: DocumentSet) -> None:
    docs_meta = docs.read_metadata()
    # Google GenAI theoretically allows up to 100 texts per call.
    # On the free tier, we can only make 100 requests per minute,
    # which would be a single call, but also 30K tokens per minute,
    # which we seem to hit constantly.
    # On the paid tier, we get server errors very frequently.
    # The larger the batch, the more likely we hit 429 rate limits
    # or 500 server errors.
    # As a result, it takes 30 minutes to index 2800 documents.
    batch_size = 1
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
    try:
        result = google_client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [e.values for e in result.embeddings]
    except ClientError:
        print("Rate limit hit, waiting for 60 seconds before retrying...", file=sys.stderr)
        time.sleep(60)
        return embedding(texts)
    except ServerError:
        print("Server error hit, waiting for 60 seconds before retrying...", file=sys.stderr)
        time.sleep(60)
        return embedding(texts)


if __name__ == "__main__":
    documents_folder = sys.argv[1]
    docs = DocumentSet(documents_folder)
    index(docs)
