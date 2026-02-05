# Makefile for Mayo Clinic STRIP AI
# Common tasks for development, testing, and deployment

# Python command detection
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
PIP := $(shell which pip3 2>/dev/null || which pip 2>/dev/null)

.PHONY: help install test lint format clean train evaluate deploy shutdown export

# Default target
help:
	@echo "Mayo Clinic STRIP AI - Available Commands:"
	@echo "=========================================="
	@echo "Development:"
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev      - Install with dev dependencies"
	@echo "  make lint             - Run linters (flake8, mypy)"
	@echo "  make format           - Format code (black, isort)"
	@echo "  make pre-commit       - Set up pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-quick       - Run quick tests only"
	@echo "  make test-cov         - Run tests with coverage"
	@echo ""
	@echo "Training:"
	@echo "  make train            - Train model with default config"
	@echo "  make train-dist       - Train with distributed (multi-GPU)"
	@echo "  make evaluate         - Evaluate trained model"
	@echo "  make compare          - Compare multiple models"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-docker    - Deploy with Docker stack"
	@echo "  make deploy-local     - Deploy locally"
	@echo "  make shutdown         - Shutdown deployment"
	@echo "  make export           - Export model to ONNX/TorchScript"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Clean generated files"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-test      - Test in Docker container"
	@echo "=========================================="

# Installation
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

# Code quality
lint:
	@echo "Running flake8..."
	flake8 src/ scripts/ deploy/ tests/ --max-line-length=120 --extend-ignore=E203,W503
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports --no-strict-optional || true

format:
	@echo "Running black..."
	black src/ scripts/ deploy/ tests/
	@echo "Running isort..."
	isort src/ scripts/ deploy/ tests/

pre-commit:
	pre-commit install
	@echo "✓ Pre-commit hooks installed"

# Testing
test:
	pytest tests/ -v --tb=short

test-quick:
	pytest tests/ -v --tb=short -k "not slow"

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term

test-enhancements:
	$(PYTHON) tests/test_enhancements.py

# Training
train:
	$(PYTHON) train.py --config config/train_config.yaml

train-dist:
	@echo "Training with 4 GPUs (adjust --nproc_per_node as needed)..."
	torchrun --nproc_per_node=4 scripts/train_distributed.py --config config/train_config.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py \
		--checkpoint experiments/best_model.pth \
		--data-dir data/processed \
		--output-dir results

compare:
	$(PYTHON) scripts/compare_models.py \
		--models resnet18 resnet50 efficientnet_b0 \
		--data-dir data/processed \
		--output results/comparison

# Deployment
deploy-docker:
	@echo "Deploying with Docker..."
	cd deploy && ./deploy.sh docker production

deploy-local:
	@echo "Deploying locally..."
	cd deploy && ./deploy.sh local development

shutdown:
	@echo "Shutting down deployment..."
	cd deploy && ./shutdown.sh

export:
	$(PYTHON) scripts/export_model.py \
		--checkpoint experiments/best_model.pth \
		--output-dir exports \
		--formats onnx torchscript

# Docker operations
docker-build:
	docker build -t mayo-strip-ai:latest -f deploy/Dockerfile .

docker-test:
	docker run --rm mayo-strip-ai:latest pytest tests/test_everything.py -v

docker-run:
	docker run -p 5000:5000 \
		-v $(PWD)/models:/app/models \
		mayo-strip-ai:latest \
		python deploy/api_with_metrics.py --checkpoint /app/models/best_model.pth

# Utilities
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf experiments/*/checkpoints/*.pth
	@echo "✓ Cleaned"

clean-all: clean
	rm -rf venv/ mlruns/ data/processed/ experiments/ exports/
	@echo "✓ Deep cleaned (removed all generated data)"

# Data preparation
prepare-data:
	$(PYTHON) scripts/prepare_data.py --data-dir data/raw --output-dir data/processed

# MLflow
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5001

# Monitoring
logs:
	@if [ -d "deploy/logs" ]; then \
		tail -f deploy/logs/*.log; \
	else \
		echo "No logs directory found"; \
	fi

docker-logs:
	cd deploy && docker-compose -f docker-compose-full.yml logs -f

# Quick development cycle
dev: format lint test
	@echo "✓ Development cycle complete"

# Full CI cycle
ci: lint test-cov
	@echo "✓ CI checks complete"
