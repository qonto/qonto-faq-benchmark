import os
import sys
import json
import random
import tqdm
from mistralai.client import Mistral
from typing import Any

from dataset.documents import DocumentSet

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium-latest"

client = Mistral(api_key=api_key)


def generate_qa_data(size: int, dataset_path: str, docs: DocumentSet) -> None:
    # Get all articles and shuffle them
    metadata_json = list(docs.read_metadata())
    random.shuffle(metadata_json)

    # If the size is larger than the number of documents, we take all documents.
    if size > len(metadata_json):
        size = len(metadata_json)
        print(f"Warning: size is larger than the number of documents. Using {size} documents.")

    # Open a file (that may not exist) in write mode:
    with open(dataset_path, "w") as f:
        # Generate `size` question/answer pairs with progress logging
        for metadata in tqdm.tqdm(metadata_json[:size], desc="Generating QA pairs"):
            # 1. Fetch the content of the article from dataset/documents/markdown/<id>.md
            article_id = metadata['id']
            with open(f"dataset/documents/markdown/{article_id}.md", "r") as article_file:
                faq = article_file.read()
            # 2. Generate a random question.
            qa = generate_qa_triplet(article_id, faq)
            # 3. Write the question/answer pair to the file.
            f.write(json.dumps(qa) + "\n")


def generate_qa_triplet(id: str, faq: str) -> dict[str, Any]:
    # 1. Generate a question to the FAQ.
    question = generate_question(faq)
    # 2. Generate an answer to the question generated in step 2.
    answer = generate_answer(faq, question)
    return {"query": question, "answer": answer, "relevant_docs": [{"id": id}]}


def generate_question(faq: str) -> str:
    prompt = "Generate a question based on the FAQ article provided. To generate it, follow these rules:\n\n" \
            "- You need to be able to answer the question from the FAQ.\n" \
            "- Write it in the language that the FAQ is written in.\n" \
            "- Your answer should only contain the question and nothing.\n" \
            "- The question should be concise.\n" \
            "- The question should not be too close to the title of the FAQ itself.\n" \
            "- Put yourself in the shoes of a customer using the app: the question must be something that a customer would ask, as flawed, ambiguous and unaware of banking systems as they would be.\n\n" \
            "You will find below the content of the FAQ article:\n\n" \
            f"<faq>\n{faq}\n</faq>"
    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content


def generate_answer(faq: str, question: str) -> str:
    prompt = "Generate an answer to the question based on the FAQ provided. To generate it, follow these rules:\n\n" \
            "- Write it in the language that the question is written in.\n" \
            "- You should only return the answer and nothing else.\n" \
            "- Address the question directly. Don't assume the user's situation without a conditional: “If you are doing this, ...”.\n" \
            "- Act as a superhuman customer support experience. The answer should be explanatory and clear.\n\n" \
            "You will find below the content of the FAQ:\n\n" \
            f"<faq>\n{faq}\n</faq>" \
            "\n\nYou will find below the question asked:\n\n" \
            f"<question>\n{question}\n</question>"
    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content


if __name__ == "__main__":
    try:
        documents_folder = sys.argv[1]
    except IndexError:
        documents_folder = "./dataset/documents"
    docs = DocumentSet(documents_folder)
    generate_qa_data(1000, "./dataset/qa/benchmark.jsonl", docs)
