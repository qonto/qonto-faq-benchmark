# Index the dataset/documents and output the index in stdout.
#
# python retriever/embeddinggemma/index.py <documents> >index.jsonl
#
# The documents must be a folder in the same format as dataset/documents.
# Format: for each document, `{id, vector}`.

import sys
import json
from itertools import islice
from typing import Any, Sequence

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from dataset.documents import DocumentSet

MODEL_ID = "google/embeddinggemma-300m"
_model = SentenceTransformer(MODEL_ID)


def index(docs: DocumentSet) -> None:
    docs_meta = docs.read_metadata()
    batch_size = 32
    for batch in tqdm(
        iter(lambda: list(islice(docs_meta, batch_size)), []),
        desc=f"Indexing documents in {docs.documents_folder}",
    ):
        index_doc_batch(batch, docs)


def index_doc_batch(docs_batch: list[dict[str, Any]], docs: DocumentSet) -> None:
    md_contents = [docs.get_from_meta(m) for m in docs_batch]
    embeddings = embed_documents(md_contents)

    for meta, vector in zip(docs_batch, embeddings):
        index_record = {"id": meta["id"], "vector": vector}
        print(json.dumps(index_record))


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Take a list of texts to index and return their embeddings."""
    embeddings = _model.encode_document(
        texts,
        batch_size=32,
        normalize_embeddings=True,
    )
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(map(float, emb)) for emb in embeddings]  # Fallback if ndarray missing tolist.


if __name__ == "__main__":
    documents_folder = sys.argv[1]
    docs = DocumentSet(documents_folder)
    index(docs)
