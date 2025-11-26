from dataset.documents import DocumentSet
from retriever.embeddinggemma.query import query as embeddinggemma_query
from benchmark.utils import benchmark


def relevant_documents_func() -> callable:
    """Returns a function that takes a question and returns a list of relevant documents."""
    dataset = DocumentSet("./dataset/documents")

    def relevant_documents(question: str, n_docs: int) -> list[str]:
        """Returns a list of relevant documents for the question."""
        index_path = "./retriever/embeddinggemma/index.jsonl"
        docs = embeddinggemma_query(question, index_path, n_docs)
        return [dataset.get(doc["id"]) for doc in docs]

    return relevant_documents


if __name__ == "__main__":
    benchmark(relevant_documents_func(), "Google Embedding Gemma 300M", 4)
