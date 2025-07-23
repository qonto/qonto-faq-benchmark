# QontoFAQ: A Retrieval Benchmark

Benchmarking embedding models with a realistic dataset
targeted at use-cases involving customer support chatbots using RAG pipelines.

## Commands

- `make dataset/documents` will download the Qonto FAQ dataset
  into a local set of cleaned up files, layed out in this way:
  - `dataset/documents/metadata.jsonl` contains `{id, path, url, title}`.
  - `dataset/documents/markdown/<id>.md` contains the Markdown content of the document.
  - `dataset/documents/raw/<id>.html` contains the raw content of the document.
- `make dataset/qa` will build a training dataset
  with query / answer pairs based on the dataset/documents,
  and store it in `dataset/qa/benchmark.jsonl`
  with each line having `{query, answer, relevant_docs: [{id}]}`.
  If dataset/documents does not exist, it downloads the Qonto FAQ by default.
- `make benchmark` will run the benchmark defined in dataset/qa/
  on all approaches,
  and output the comparative results in `benchmark/result.json`.
- `make benchmark/<approach>` will run the benchmark on a given algorithm,
  and output `benchmark/<approach>/result.json`,
  along with `benchmark/<approach>/log.txt` for investigations.
- `python -m retriever.<approach>.index [documents] >index.jsonl`
  will index the dataset/documents and output the index in stdout.
  The documents must be a folder in the same format as dataset/documents.
  Format: for each document, `{id, vector}`.
  You may need to `source ./env` to have the right environment variables.
- `python -m retriever.<approach>.query --index [index.jsonl] [query] >result.json`
  will query the index with the given query
  and output a list of relevant documents as JSON through stdout.
  Format: list of documents `[{id, score, snippet}]`.

## Requirements

- huggingface-cli
- Have in your environment:
  - `OPENAI_API_KEY` from <https://platform.openai.com/api-keys>
  - `GEMINI_API_KEY` from <https://aistudio.google.com/apikey>
  - `MISTRAL_API_KEY` from <https://console.mistral.ai/api-keys/>
  - `COHERE_API_KEY` from <https://dashboard.cohere.com/api-keys>
  - `VOYAGE_API_KEY` from <https://dashboard.voyageai.com/api-keys>
  - `JINA_API_KEY` from <https://jina.ai/api-dashboard/key-manager>
  - `DOCUMENTS_FOLDER` (if located at a non-default path)
- Using venv, run `source ./.venv/bin/activate` to activate the environment
- Preferentially, a GPU with CUDA support

## Local Qwen3 Server

To run the Qwen3 benchmark with Huggingface’s TEI:

```bash
docker run --gpus all -p 8114:80 -v hf_cache:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.7.2 --model-id Qwen/Qwen3-Embedding-8B --dtype float16
```

We also tried running it with a local llama.cpp server, with worse results, following these steps:

1.  **Download the model:**
    ```bash
    huggingface-cli download Qwen/Qwen3-Embedding-8B-GGUF Qwen3-Embedding-8B-Q8_0.gguf
    ```

2.  **Copy the model to your models directory:**
    ```bash
    cp --reflink /home/tyl/.cache/huggingface/hub/*/*/*/Qwen3-Embedding-8B-Q8_0.gguf /data/ml/models/gguf/Qwen3-Embedding-8B-Q8_0.gguf
    ```

3.  **Run the llama.cpp server using Docker:**
    ```bash
    docker run --gpus all -v /data/ml/models/gguf:/models -p 8114:8080 ghcr.io/ggml-org/llama.cpp:full-cuda -s --host 0.0.0.0 -m /models/Qwen3-Embedding-8B-Q8_0.gguf --embedding --pooling last -c 32768 -ub 8192 --verbose-prompt --n-gpu-layers 999
    ```

The worse results may have been caused by [incomplete llama.cpp support][llama-cpp-qwen3].

[llama-cpp-qwen3]: https://github.com/ggml-org/llama.cpp/pull/14029
