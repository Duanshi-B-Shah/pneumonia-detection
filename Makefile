.PHONY: setup download split train eval serve demo test lint docker clean help

PYTHON := python3
PIP := pip

# ─── Setup ────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install all dependencies (editable mode)
	$(PIP) install -e ".[all]"
	pre-commit install

setup-minimal: ## Install core dependencies only
	$(PIP) install -e .

# ─── Data ─────────────────────────────────────────────────────────────
download: ## Download dataset from Kaggle
	bash scripts/download_data.sh

split: ## Re-split dataset (stratified 80/10/10)
	$(PYTHON) scripts/split_data.py \
		--input data/raw/chest_xray \
		--output data/processed \
		--train-ratio 0.8 \
		--val-ratio 0.1 \
		--seed 42

# ─── Training ─────────────────────────────────────────────────────────
train: ## Train the model
	$(PYTHON) -m pneumonia.training.trainer --config configs/train_config.yaml

eval: ## Evaluate model on test set
	$(PYTHON) -m pneumonia.training.evaluator --config configs/train_config.yaml

# ─── Serving ──────────────────────────────────────────────────────────
serve: ## Start FastAPI server
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

demo: ## Start Gradio demo UI
	$(PYTHON) ui/demo.py

# ─── Export ───────────────────────────────────────────────────────────
export-onnx: ## Export model to ONNX format
	$(PYTHON) scripts/export_onnx.py \
		--checkpoint checkpoints/best_model.pth \
		--output checkpoints/model.onnx

# ─── Quality ──────────────────────────────────────────────────────────
test: ## Run tests
	pytest

lint: ## Run linter
	ruff check src/ api/ tests/
	ruff format --check src/ api/ tests/

format: ## Auto-format code
	ruff check --fix src/ api/ tests/
	ruff format src/ api/ tests/

# ─── Docker ───────────────────────────────────────────────────────────
docker: ## Build Docker image
	docker build -t pneumonia-detection:latest .

docker-run: ## Run Docker container
	docker-compose up -d

docker-stop: ## Stop Docker container
	docker-compose down

# ─── Cleanup ──────────────────────────────────────────────────────────
clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
