# Aditya-FlareCast — developer task runner
# Usage: `make help`

PY ?= python
CONFIG ?= configs/default.yaml

.DEFAULT_GOAL := help
.PHONY: help install install-dev synth preprocess nowcast train pipeline \
        predict report test lint format serve dashboard docker-build \
        docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the core package (editable)
	pip install -e .

install-dev: ## Install with all optional extras + dev tools
	pip install -e ".[all,dev]"

synth: ## Generate synthetic SoLEXS + HEL1OS data
	$(PY) -m aditya_flarecast.cli synth --config $(CONFIG)

preprocess: ## Fuse + clean raw light curves
	$(PY) -m aditya_flarecast.cli preprocess --config $(CONFIG)

nowcast: ## Detect flares -> master catalogue + SQLite DB
	$(PY) -m aditya_flarecast.cli nowcast --config $(CONFIG)

train: ## Train the forecaster
	$(PY) -m aditya_flarecast.cli train --config $(CONFIG)

pipeline: ## Run the entire pipeline end-to-end (synthetic data)
	$(PY) -m aditya_flarecast.cli pipeline --config $(CONFIG)

predict: ## Print the latest live forecast
	$(PY) -m aditya_flarecast.cli predict --config $(CONFIG)

report: ## Generate evaluation plots + metrics report
	$(PY) scripts/generate_report.py --config $(CONFIG)

test: ## Run the test suite
	$(PY) -m pytest -q

lint: ## Lint with ruff
	ruff check src tests

format: ## Auto-format with black + ruff --fix
	black src tests && ruff check --fix src tests

serve: ## Launch the FastAPI service
	$(PY) -m aditya_flarecast.cli serve-api --config $(CONFIG)

dashboard: ## Launch the Streamlit dashboard
	$(PY) -m aditya_flarecast.cli dashboard --config $(CONFIG)

docker-build: ## Build the Docker image
	docker build -t aditya-flarecast:latest .

docker-up: ## Run API + dashboard via docker-compose
	docker compose up --build

clean: ## Remove generated data, models, and caches
	rm -rf data/raw/* data/interim/* data/processed/* data/catalogues/* models/* \
		reports/*.png reports/*.pdf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
