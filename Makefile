.PHONY: setup run lint test help

# Python and executable paths using the venv
PYTHON = venv\Scripts\python
UVICORN = venv\Scripts\uvicorn
RUFF = venv\Scripts\ruff
PYTEST = venv\Scripts\pytest

## Default target
help:
	@echo "Available commands:"
	@echo "  make setup   - Create venv and install all dependencies"
	@echo "  make run     - Start the development server on port 8001"
	@echo "  make lint    - Run Ruff linter"
	@echo "  make test    - Run pytest"

## Create venv and install dependencies
setup:
	python -m venv venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. cp .env.example .env"
	@echo "  2. Fill in your AWS credentials and BEDROCK_KNOWLEDGE_BASE_ID"
	@echo "  3. make run"

## Start the FastAPI dev server
run:
	$(UVICORN) main:app --reload --port 8001

## Lint with Ruff
lint:
	$(RUFF) check .
	$(RUFF) format --check .

## Run tests
test:
	$(PYTEST) tests/ -v
