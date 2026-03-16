PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup test run-demo run-ollama clean lint

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest

run-demo:
	$(PY) -m src.cli --input examples/lemonade_plan_missing_juicing.txt --runs-dir runs --backend demo

run-ollama:
	$(PY) -m src.cli --input examples/lemonade_plan_missing_juicing.txt --runs-dir runs --backend ollama --ollama-model $${OLLAMA_MODEL:-llama3.1:8b}

clean:
	rm -rf runs/*
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

lint:
	$(PY) -m ruff check src/ tests/
