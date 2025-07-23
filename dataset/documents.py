import os
from typing import Any, Generator

from dataset.jsonl import parse_jsonl

class DocumentSet():
    def __init__(self, documents_folder: str):
        self.documents_folder = documents_folder

    def read_metadata(self) -> Generator[dict[str, Any], None, None]:
        docs_meta_path = os.path.join(self.documents_folder, "metadata.jsonl")
        return parse_jsonl(docs_meta_path)

    def get_from_meta(self, meta: dict[str, Any]) -> str:
        with open(meta["path"], "r") as f:
            return f.read()

    def get(self, doc_id: str) -> str:
        path = os.path.join(self.documents_folder, "markdown", f"{doc_id}.md")
        with open(path, "r") as f:
            return f.read()

    def __getitem__(self, doc_id: str) -> str:
        return self.get(doc_id)
