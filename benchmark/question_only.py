from benchmark.utils import benchmark

def no_documents(question: str, n_docs: int) -> list[str]:
    return []

if __name__ == "__main__":
    benchmark(no_documents, 1)
