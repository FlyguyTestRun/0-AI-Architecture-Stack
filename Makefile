# Zerostack developer commands.
#
# The only prerequisite for `make setup` and `make demo` is Python 3.11 or newer.
# Docker is needed only for `make up`.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
MODEL  ?= gemma3:4b

.DEFAULT_GOAL := help
.PHONY: help setup demo doctor ingest serve ui test lint fmt up down logs pull-model clean check evals

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip

setup: $(BIN)/python ## Create the virtualenv and install the package
	$(BIN)/pip install --quiet -e ".[dev]"
	@echo "Setup complete. Run 'make demo' to see the stack work."

demo: setup ## Ingest the sample corpus and run scripted questions
	$(BIN)/zerostack demo

doctor: setup ## Report which backend is active in every layer
	$(BIN)/zerostack doctor

ingest: setup ## Index data/corpus into the vector store
	$(BIN)/zerostack ingest

serve: setup ## Run the FastAPI service on port 8000
	$(BIN)/zerostack serve --reload

ui: setup ## Run the Streamlit frontend
	$(BIN)/pip install --quiet -e ".[ui]"
	$(BIN)/streamlit run apps/streamlit_app/app.py

test: setup ## Run the test suite
	$(BIN)/pytest -q

lint: setup ## Check formatting, lint rules and the em dash policy
	$(BIN)/ruff check src tests apps
	$(BIN)/ruff format --check src tests apps
	$(PYTHON) scripts/check_emdash.py

fmt: setup ## Apply formatting and autofixable lint rules
	$(BIN)/ruff format src tests apps
	$(BIN)/ruff check --fix src tests apps

evals: setup ## Measure retrieval and answer quality against the golden set
	$(BIN)/zerostack evaluate --min-pass-rate 1.0 --min-recall 1.0

check: lint test evals ## Everything CI runs

up: ## Start Qdrant, Ollama and Phoenix
	docker compose up -d
	@echo "Qdrant   http://localhost:6333/dashboard"
	@echo "Phoenix  http://localhost:6006"
	@echo "Next:    make pull-model"

down: ## Stop the containers
	docker compose down

logs: ## Follow container logs
	docker compose logs -f

pull-model: ## Pull the default model into Ollama
	docker compose exec ollama ollama pull $(MODEL)

clean: ## Remove the virtualenv, caches and local state
	rm -rf $(VENV) .pytest_cache .ruff_cache
	rm -f data/zerostack.db data/analytics.duckdb data/traces.jsonl
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
