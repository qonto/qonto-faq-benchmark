SHELL := /bin/bash

benchmark/plot.svg: benchmark/plot.py benchmark/*.json
	@echo "Generating benchmark plot..."
	source ./.venv/bin/activate; \
	python -m benchmark.plot

benchmark: benchmark/question_only.json benchmark/random.json benchmark/bm25.json benchmark/mistral.json benchmark/openai.json benchmark/cohere.json benchmark/google.json benchmark/voyageai.json benchmark/qwen3.json benchmark/jina.json benchmark/embeddinggemma.json benchmark/luxical.json

dataset/documents: .setup
	@if [ -d dataset/documents ]; then \
	  echo "$@ already exist. Skipping scraping FAQ."; \
	  exit 0; \
	else \
	  echo "$@ does not exist. Scraping FAQ."; \
	  mkdir -p dataset/documents; \
	  source ./.venv/bin/activate; \
	  python -m dataset.scrape_qonto_faq; \
	  touch dataset/documents; \
	fi

dataset/qa: dataset/documents
	@if [ -d dataset/qa ]; then \
	  echo "$@ already exist. Skipping generating the QA dataset."; \
	  exit 0; \
	else \
	  echo "$@ does not exist. Generating the QA dataset."; \
	  mkdir -p dataset/qa; \
	  source ~/.venv/bin/activate; \
	  python -m dataset.gen_qa_data; \
	  touch dataset/qa; \
	fi

retriever/bm25/index: dataset/documents
	@echo "Building BM25 index..."
	source ./.venv/bin/activate; \
	python -m retriever.bm25.index dataset/documents retriever/bm25/index

retriever/mistral/index.jsonl: dataset/documents
	@echo "Building Mistral index..."
	source ./.venv/bin/activate; \
	python -m retriever.mistral.index dataset/documents >retriever/mistral/index.jsonl

retriever/openai/index.jsonl: dataset/documents
	@echo "Building OpenAI index..."
	source ./.venv/bin/activate; \
	python -m retriever.openai.index dataset/documents >retriever/openai/index.jsonl

retriever/google/index.jsonl: dataset/documents
	@echo "Building Gemini index..."
	source ./.venv/bin/activate; \
	python -m retriever.google.index dataset/documents >retriever/google/index.jsonl

retriever/cohere/index.jsonl: dataset/documents
	@echo "Building Cohere index..."
	source ./.venv/bin/activate; \
	python -m retriever.cohere.index dataset/documents >retriever/cohere/index.jsonl

retriever/voyageai/index.jsonl: dataset/documents
	@echo "Building VoyageAI index..."
	source ./.venv/bin/activate; \
	python -m retriever.voyageai.index dataset/documents >retriever/voyageai/index.jsonl

retriever/qwen3/index.jsonl: dataset/documents
	@echo "Building Qwen3 index..."
	source ./.venv/bin/activate; \
	python -m retriever.qwen3.index dataset/documents >retriever/qwen3/index.jsonl

retriever/jina/index.jsonl: dataset/documents
	@echo "Building Jina index..."
	source ./.venv/bin/activate; \
	python -m retriever.jina.index dataset/documents >retriever/jina/index.jsonl

retriever/embeddinggemma/index.jsonl: dataset/documents
	@echo "Building EmbeddingGemma index..."
	source ./.venv/bin/activate; \
	python -m retriever.embeddinggemma.index dataset/documents >retriever/embeddinggemma/index.jsonl

retriever/luxical/index.jsonl: dataset/documents
	@echo "Building Luxical index..."
	source ./.venv/bin/activate; \
	python -m retriever.luxical.index dataset/documents >retriever/luxical/index.jsonl

benchmark/question_only.json: dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.question_only >benchmark/question_only.json
	cat benchmark/question_only.json

benchmark/random.json: dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.random >benchmark/random.json
	cat benchmark/random.json

benchmark/bm25.json: retriever/bm25/index dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.bm25 >benchmark/bm25.json
	cat benchmark/bm25.json

benchmark/mistral.json: retriever/mistral/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.mistral >benchmark/mistral.json
	cat benchmark/mistral.json

benchmark/openai.json: retriever/openai/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.openai >benchmark/openai.json
	cat benchmark/openai.json

benchmark/google.json: retriever/google/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.google >benchmark/google.json
	cat benchmark/google.json

benchmark/cohere.json: retriever/cohere/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.cohere >benchmark/cohere.json
	cat benchmark/cohere.json

benchmark/voyageai.json: retriever/voyageai/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.voyageai >benchmark/voyageai.json
	cat benchmark/voyageai.json

benchmark/qwen3.json: retriever/qwen3/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.qwen3 >benchmark/qwen3.json
	cat benchmark/qwen3.json

benchmark/jina.json: retriever/jina/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.jina >benchmark/jina.json
	cat benchmark/jina.json

benchmark/embeddinggemma.json: retriever/embeddinggemma/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.embeddinggemma >benchmark/embeddinggemma.json
	cat benchmark/embeddinggemma.json

benchmark/luxical.json: retriever/luxical/index.jsonl dataset/qa
	mkdir -p benchmark
	source ./.venv/bin/activate; \
	python -m benchmark.luxical >benchmark/luxical.json
	cat benchmark/luxical.json

.setup:
	python -m venv .venv
	source ./.venv/bin/activate; \
	pip install -r requirements.txt
	touch .setup
