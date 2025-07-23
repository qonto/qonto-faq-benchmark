# Benchmark a random sampler of the dataset/documents
# using the dataset/qa/benchmark.jsonl to estimate information gain.

import random
from dataset.documents import DocumentSet
from benchmark.utils import benchmark


def relevant_documents_func(n_docs: int, dataset: DocumentSet, metadata: list[dict]) -> list[str]:
    """Returns a function that takes a question and returns a list of relevant documents."""
    def relevant_documents(question: str) -> list[str]:
        """Returns a list of relevant documents for the question."""
        # Fetch "relevant" documents.
        docs = random_sample(n_docs, metadata)
        return [dataset.get(d['id']) for d in docs]
    return relevant_documents


def random_sample(n_docs: int, metadata: list[dict]) -> list[dict]:
    """Returns a random sample of n_docs from the documents metadata."""
    sampled_docs = random.sample(metadata, n_docs)
    return sampled_docs


if __name__ == "__main__":
    dataset = DocumentSet("./dataset/documents")
    metadata = list(dataset.read_metadata())
    for i in range(4):
        print(f"Benchmarking with {i + 1} relevant documents...")
        benchmark(relevant_documents_func(i + 1, dataset, metadata))
