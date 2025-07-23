
from dataset.documents import DocumentSet
from retriever.jina.query import query as jina_query
from benchmark.utils import benchmark


def relevant_documents_func(n_docs: int, dataset: DocumentSet) -> list[str]:
    """Returns a function that takes a question and returns a list of relevant documents."""
    def relevant_documents(question: str) -> list[str]:
        """Returns a list of relevant documents for the question."""
        # Fetch relevant documents.
        jina_index = "./retriever/jina/index.jsonl"
        docs = jina_query(question, jina_index, n_docs)
        return [dataset.get(d['id']) for d in docs]
    return relevant_documents


if __name__ == "__main__":
    dataset = DocumentSet("./dataset/documents")
    for i in range(4):
        print(f"Benchmarking with {i + 1} relevant documents...")
        benchmark(relevant_documents_func(i + 1, dataset))
