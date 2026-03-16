.PHONY: setup test run-demo run-ollama clean lint

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

run-demo:
	python -m src --input examples/lemonade_plan_missing_juicing.txt --runs-dir runs --backend demo

run-ollama:
	python -m src --input examples/lemonade_plan_missing_juicing.txt --runs-dir runs --backend ollama --ollama-base-url http://127.0.0.1:11434 --ollama-model llama3.1:8b-instruct-q4_K_M --ollama-max-retries 2

clean:
	rm -rf runs/*
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

lint:
	ruff check src/ tests/
