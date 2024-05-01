.PHONY: install update help

install: ## Install the virtual environment and dependencies.
	@echo "📦 Creating virtual environment"
	@python3 -m venv .venv && \
	echo "Installing dependencies from pyproject.toml..." && \
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e .

update: ## Update the dependencies.
	@echo "📦 Updating dependencies"
	@. .venv/bin/activate && \
	pip install --upgrade pip && \
	echo "Exporting installed packages to requirements.txt" && \
	pip freeze >> requirements.txt
	@echo "Dependencies updated!"

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help