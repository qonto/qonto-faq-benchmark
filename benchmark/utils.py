import os
import json
import time
from tqdm import tqdm
from typing import Any, Callable
import requests
import math
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataset.jsonl import parse_jsonl


def benchmark(relevant_documents: Callable[str, list[str]], max_docs: int = 4) -> None:
    """
    Benchmark the dataset in ./dataset/qa/benchmark.jsonl.
    Prints out the result in stdout, as a JSON object
    with the following structure:

    {
        "model": "model_name",
        "measurements": [
            {"top_k": 1, "information_assimilation": 0.85},
            {"top_k": 2, "information_assimilation": 0.90},
            ...
        ]
    }

    relevant_documents: a function that takes a question (str) and returns a
        list of relevant documents (as a list of strings).
    max_docs: we will benchmark using 1 document, 2, ... up to max_docs.
    """
    # `dataset/qa/benchmark.jsonl` contains lines of the form `{query, answer}`.
    # Let's read it record by record.
    validation_set = list(parse_jsonl("./dataset/qa/benchmark.jsonl"))

    # Get the relevant documents for each question in the Question/Answer dataset.
    relevant_docs = {}
    for data in tqdm(validation_set, desc="Fetching relevant documents"):
        query = data["query"]
        relevant_docs[query] = relevant_documents(query, max_docs)
    torch.cuda.empty_cache()

    tokenizer, model = load_model()
    model_name = model.name_or_path

    # Pre-compute docless values.
    n_bits_docless, n_bytes = answer_info_with_n_docs(
        validation_set, relevant_docs, 0, tokenizer, model
    )

    compression_ratio_docless = n_bits_docless / (n_bytes * 8)

    measurements = []
    for i in range(max_docs):
        n_docs = i + 1
        n_bits, n_bytes = answer_info_with_n_docs(
            validation_set, relevant_docs, n_docs, tokenizer, model
        )
        compression_ratio = n_bits / (n_bytes * 8)
        score = 1 - (compression_ratio / compression_ratio_docless)
        measurements.append({"top_k": n_docs, "information_assimilation": score})

    result = {"model": model_name, "measurements": measurements}
    print(json.dumps(result, indent=4))


def answer_info_with_n_docs(
    validation_set: list[dict[str, Any]],
    relevant_docs: dict[str, list[str]],
    n_docs: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> tuple[float, int]:
    """
    Computes the sum of bits and bytes for the answers in the validation set,
    given n_docs relevant documents.
    """
    n_bits = 0
    n_bytes = 0
    desc = (
        "Estimating baseline (doc-less) compression ratio"
        if n_docs == 0
        else f"Estimating compression ratio with {n_docs} docs"
    )
    for data in tqdm(validation_set, desc=desc):
        query = data["query"]
        answer = data["answer"]
        docs = relevant_docs[query][:n_docs]
        tokens, logprobs = transformers_answer_probs(docs, query, answer, tokenizer, model)
        qa_n_bits, qa_n_bytes = answer_compression(tokens, logprobs)
        n_bits += qa_n_bits
        n_bytes += qa_n_bytes
        torch.cuda.empty_cache()
    return n_bits, n_bytes


def load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    #model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # Out of memory w/ Refran on 24GB GPU:
    #model_name = "google/gemma-3-1b-it"
    #model_name = "google/gemma-3-4b-it"
    #model_name = "microsoft/Phi-4-mini-instruct"
    #model_name = "mistralai/Ministral-8B-Instruct-2410"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device=torch_device)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()
    return tokenizer, model


# On 100 iterations, it takes ~12 seconds with CUDA.
def transformers_answer_probs(docs: list[str], question: str, answer: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> tuple[list[str], list[float]]:
    """Returns (tokens, logprobs) for each token in the answer."""
    with torch.inference_mode():
        pt_tokens = tokenizer.apply_chat_template([
            # Warning: Ministral does not have system prompts, and will silently discard them.
            {"role": "system", "content": system_prompt(docs)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
            tokenizer=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device.type)
        # We take batch 0 of the tokens' input_ids.
        all_token_ids: list[int] = [tok_id.item() for tok_id in pt_tokens.input_ids[0]]
        all_tokens: list[str] = [tokenizer.decode(tok_id) for tok_id in all_token_ids]
        # We only need to compute the probabilities for the tokens in the answer.
        start_idx = assistant_start_idx(all_tokens, model.name_or_path)
        answer_token_ids = all_token_ids[start_idx:]
        answer_tokens = all_tokens[start_idx:]
        # Type: {logits: batch[tokens[vocab[]]], past_key_values}
        pred_batch = model(**pt_tokens)
        # all_tokens is (a, b, c, d), where (c, d) is the answer, and stard_idx is on c.
        # pred_batch is (b, c, d, e). So we take start_idx-1 to -1.
        prob_batch = softmax(pred_batch.logits[:, start_idx-1:-1, :], dim=-1)
        # Probabilities of each token in the sequence (both question and answer).
        answer_logprobs: list[float] = [log2(prob_batch[0][i][answer_token_ids[i]])
            for i in range(len(prob_batch[0]))]
        assert len(answer_tokens) == len(answer_logprobs), \
            f"Answer tokens ({len(answer_tokens)}) and logprobs ({len(answer_logprobs)}) mismatch."
        return answer_tokens, answer_logprobs


# On 100 iterations, it takes from 3 minutes to 114 minutes.
def together_answer_probs(docs: list[str], question: str, answer: str) -> tuple[list[str], list[float]]:
    """Returns (tokens, logprobs) for each token in the answer."""

    resp = requests.post("https://api.together.xyz/v1/chat/completions", json={
        "messages": [
            { "role": "system", "content": system_prompt(docs) },
            { "role": "user", "content": question },
            { "role": "assistant", "content": answer },
        ],
        "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "logprobs": 1,
        "echo": True
    }, headers={
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
    }).json()

    while "error" in resp:
        if 'type' in resp['error'] and 'credit_limit' == resp['error']['type']:
            # We reached rate limits. Let's wait for a second and retry.
            time.sleep(1)
            return together_answer_probs(question, answer)
        elif 'Server is overloaded' in resp['error']['message']:
            # Overloads take more time to recover from. Wait a minute.
            time.sleep(60)
            return together_answer_probs(question, answer)
        else:
            raise Exception(f"Together AI error: {resp}")

    if "prompt" not in resp:
        raise Exception(f"Unexpected response: {resp}")

    # Together AI yields the full list of tokens, when using echo=True.
    # So we need to find the start of the assistant's response.
    all_tokens = resp["prompt"][0]["logprobs"]["tokens"]
    all_logprobs = resp["prompt"][0]["logprobs"]["token_logprobs"]
    return assistant_response(all_tokens, all_logprobs)


def system_prompt(docs: list[str]) -> str:
    return "You are a helpful AI agent. The user will ask a question.\n" + \
        "Here are relevant documents related to that question, " + \
        "to help you answer accurately:\n\n" + \
        "\n\n".join([f"<document>\n{doc}\n</document>" for doc in docs])


def assistant_response(tokens: list[str], logprobs: list[float], model_name: str) -> tuple[list[str], list[float]]:
    """Takes a list of tokens and matching logprobs containing both user and assistant messages.
    Returns only the assistant's response and its logprobs."""
    start_idx = assistant_start_idx(tokens, model_name)
    return tokens[start_idx:], logprobs[start_idx:]


def assistant_start_idx(tokens: list[str], model_name: str) -> int:
    """Returns the index of the first token of the assistant's response."""
    # Find the start of the assistant's response.
    for i in range(len(tokens)):
        if model_name == "meta-llama/Llama-3.2-1B-Instruct" and tokens[i] == "assistant" and tokens[i+1] == "<|end_header_id|>":
            # There is also a \n\n token to ignore after the end header.
            start_idx = i + 3
            break
        if "gemma-3" in model_name and tokens[i] == "<start_of_turn>" and tokens[i+1] == "model":
            # There is also a \n token to ignore after the end header.
            start_idx = i + 3
            break
        if "SmolLM2" in model_name and tokens[i] == "<|im_start|>" and tokens[i+1] == "ass" and tokens[i+2] == "istant":
            # There is also a \n token to ignore after the end header.
            start_idx = i + 4
            break
        if model_name == "mistralai/Ministral-8B-Instruct-2410" and tokens[i] == "[/INST]":
            start_idx = i + 1
            break
        if model_name == "microsoft/Phi-4-mini-instruct" in model_name and tokens[i] == "<|assistant|>":
            start_idx = i + 1
            break
    return start_idx


def answer_compression(tokens: list[str], logprobs: list[float]) -> tuple[float, int]:
    """Returns (information content in bits, number of bytes)"""
    n_bytes = sum([len(token.encode("utf-8")) for token in tokens])
    return -sum(logprobs), n_bytes


def log2(value: float) -> float:
    return math.log2(value) if value != 0 else -math.inf
