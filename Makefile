SHELL := /bin/bash

dataset/documents: .setup
	mkdir -p dataset/documents
	source ./.venv/bin/activate; \
	python -m dataset.scrape_qonto_faq
	touch dataset/documents

dataset/qa: dataset/documents
	mkdir -p dataset/qa
	source ~/.venv/bin/activate; \
	python -m dataset.gen_qa_data
	touch dataset/qa

retriever/bm25/index: dataset/documents
	source ./.venv/bin/activate; \
	python -m retriever.bm25.index dataset/documents retriever/bm25/index

retriever/mistral/index.jsonl: dataset/documents
	source ./.venv/bin/activate; \
	python -m retriever.mistral.index dataset/documents >retriever/mistral/index.jsonl

retriever/openai/index.jsonl: dataset/documents
	source ./.venv/bin/activate; \
	python -m retriever.openai.index dataset/documents >retriever/openai/index.jsonl

retriever/google/index.jsonl: dataset/documents
	source ./.venv/bin/activate; \
	python -m retriever.google.index dataset/documents >retriever/google/index.jsonl

retriever/cohere/index.jsonl: dataset/documents
	source ./.venv/bin/activate; \
	python -m retriever.cohere.index dataset/documents >retriever/cohere/index.jsonl

retriever/voyageai/index.jsonl: dataset/documents
	source ./.venv/bin/activate; \
	python -m retriever.voyageai.index dataset/documents >retriever/voyageai/index.jsonl

retriever/qwen3/index.jsonl: dataset/documents
	source ./.venv/bin/activate; \
	python -m retriever.qwen3.index dataset/documents >retriever/qwen3/index.jsonl

retriever/jina/index.jsonl: dataset/documents
	source ./.venv/bin/activate; \
	python -m retriever.jina.index dataset/documents >retriever/jina/index.jsonl

benchmark: benchmark/question_only benchmark/random benchmark/bm25 benchmark/mistral benchmark/openai benchmark/cohere benchmark/google benchmark/voyageai benchmark/qwen3 benchmark/jina

benchmark/question_only: dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.question_only >benchmark/question_only
	cat benchmark/question_only

benchmark/random: dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.random >benchmark/random
	cat benchmark/random

benchmark/bm25: retriever/bm25/index dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.bm25 >benchmark/bm25
	cat benchmark/bm25

benchmark/mistral: retriever/mistral/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.mistral >benchmark/mistral
	cat benchmark/mistral

benchmark/openai: retriever/openai/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.openai >benchmark/openai
	cat benchmark/openai

benchmark/google: retriever/google/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.google >benchmark/google
	cat benchmark/google

benchmark/cohere: retriever/cohere/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.cohere >benchmark/cohere
	cat benchmark/cohere

benchmark/voyageai: retriever/voyageai/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.voyageai >benchmark/voyageai
	cat benchmark/voyageai

benchmark/qwen3: retriever/qwen3/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.qwen3 >benchmark/qwen3
	cat benchmark/qwen3

benchmark/jina: retriever/jina/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.jina >benchmark/jina
	cat benchmark/jina

.setup:
	python -m venv .venv
	source ./.venv/bin/activate; \
	pip install -r requirements.txt
	touch .setup

.PHONY: benchmark/question_only
