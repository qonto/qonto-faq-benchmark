from dataset.documents import DocumentSet
from retriever.qwen3.query import query as qwen3_query
from benchmark.utils import benchmark


def relevant_documents_func() -> callable:
    """Returns a function that takes a question and returns a list of relevant documents."""
    dataset = DocumentSet("./dataset/documents")
    def relevant_documents(question: str, n_docs: int) -> list[str]:
        """Returns a list of relevant documents for the question."""
        # Fetch relevant documents.
        qwen3_index = "./retriever/qwen3/index.jsonl"
        docs = qwen3_query(question, qwen3_index, n_docs)
        return [dataset.get(d['id']) for d in docs]
    return relevant_documents


if __name__ == "__main__":
    benchmark(relevant_documents_func(), "Qwen 3 Embedding 8B", 4)
