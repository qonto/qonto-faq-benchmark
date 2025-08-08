from dataset.documents import DocumentSet
from retriever.openai.query import query as openai_query
from benchmark.utils import benchmark


def relevant_documents_func() -> callable:
    """Returns a function that takes a question and returns a list of relevant documents."""
    dataset = DocumentSet("./dataset/documents")
    def relevant_documents(question: str, n_docs: int) -> list[str]:
        """Returns a list of relevant documents for the question."""
        # Fetch relevant documents.
        openai_index = "./retriever/openai/index.jsonl"
        docs = openai_query(question, openai_index, n_docs)
        return [dataset.get(d['id']) for d in docs]
    return relevant_documents


if __name__ == "__main__":
    benchmark(relevant_documents_func(), "OpenAI text-embedding-3-large", 4)
