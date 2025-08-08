from dataset.documents import DocumentSet
from retriever.cohere.query import query as cohere_query
from benchmark.utils import benchmark


def relevant_documents_func() -> callable:
    """Returns a function that takes a question and returns a list of relevant documents."""
    dataset = DocumentSet("./dataset/documents")
    def relevant_documents(question: str, n_docs: int) -> list[str]:
        """Returns a list of relevant documents for the question."""
        # Fetch relevant documents.
        cohere_index = "./retriever/cohere/index.jsonl"
        docs = cohere_query(question, cohere_index, n_docs)
        return [dataset.get(d['id']) for d in docs]
    return relevant_documents


if __name__ == "__main__":
    benchmark(relevant_documents_func(), "Cohere embed v4.0", 4)
