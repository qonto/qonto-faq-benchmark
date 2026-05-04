# Index the dataset/documents and output the index in stdout.
#
# python retriever/<approach>/index.py <documents> >index.jsonl
#
# The documents must be a folder in the same format as dataset/documents.
# Format: for each document, `{id, vector}`.

import sys
import json
import tqdm
from typing import Any
from pathlib import Path
from itertools import islice

# Add the project root to Python path to import dataset module
sys.path.append(str(Path(__file__).parent.parent.parent))

import luxical.embedder
from dataset.documents import DocumentSet

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

def index(docs: DocumentSet) -> None:
    """Index the documents using luxical embeddings."""
    # Get the model path
    model_path = get_model_path()

    # Load the luxical embedder
    print(f"Loading luxical model from: {model_path}", file=sys.stderr)
    embedder = luxical.embedder.Embedder.load(model_path)

    # Dataset handling
    docs_meta = docs.read_metadata()
    batch_size = 16

    # Process documents in batches
    for batch in tqdm.tqdm(iter(lambda: list(islice(docs_meta, batch_size)), []), desc="Indexing documents"):
        index_doc_batch(batch, docs, embedder)

def index_doc_batch(docs_batch: list[dict[str, Any]], docs: DocumentSet, embedder: luxical.embedder.Embedder) -> None:
    """Index a batch of documents."""
    # Read the markdown files
    md_contents = [docs.get_from_meta(m) for m in docs_batch]

    # Generate embeddings
    embeddings = embedder(md_contents, progress_bars=False)

    # Output the document's index information
    for i in range(len(docs_batch)):
        index = {
            "id": docs_batch[i]["id"],
            "vector": embeddings[i].tolist()  # Convert numpy array to list for JSON serialization
        }
        print(json.dumps(index))

if __name__ == "__main__":
    documents_folder = sys.argv[1]
    docs = DocumentSet(documents_folder)
    index(docs)
