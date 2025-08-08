from dataset.documents import DocumentSet
from retriever.voyageai.query import query as voyageai_query
from benchmark.utils import benchmark


def relevant_documents_func() -> callable:
    """Returns a function that takes a question and returns a list of relevant documents."""
    dataset = DocumentSet("./dataset/documents")
    def relevant_documents(question: str, n_docs: int) -> list[str]:
        """Returns a list of relevant documents for the question."""
        # Fetch relevant documents.
        voyageai_index = "./retriever/voyageai/index.jsonl"
        docs = voyageai_query(question, voyageai_index, n_docs)
        return [dataset.get(d['id']) for d in docs]
    return relevant_documents


if __name__ == "__main__":
    benchmark(relevant_documents_func(), "Voyage AI 3 Large", 4)
