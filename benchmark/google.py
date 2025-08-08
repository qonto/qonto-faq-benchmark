from dataset.documents import DocumentSet
from retriever.google.query import query as google_query
from benchmark.utils import benchmark


def relevant_documents_func() -> callable:
    """Returns a function that takes a question and returns a list of relevant documents."""
    dataset = DocumentSet("./dataset/documents")
    def relevant_documents(question: str, n_docs: int) -> list[str]:
        """Returns a list of relevant documents for the question."""
        # Fetch relevant documents.
        google_index = "./retriever/google/index.jsonl"
        docs = google_query(question, google_index, n_docs)
        return [dataset.get(d['id']) for d in docs]
    return relevant_documents


if __name__ == "__main__":
    benchmark(relevant_documents_func(), "Google gemini-embedding-001", 4)
