.PHONY: help install dev test lint format build run clean docker-build docker-run docker-stop

# Default target
help:
	@echo "Wenah Civil Rights Compliance Framework"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install      Install dependencies"
	@echo "  dev          Install with dev dependencies"
	@echo "  test         Run tests with coverage"
	@echo "  test-quick   Run tests without coverage"
	@echo "  lint         Run linters"
	@echo "  format       Format code"
	@echo ""
	@echo "Running:"
	@echo "  run          Run the API server locally"
	@echo "  run-dev      Run with hot reload"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run with Docker Compose"
	@echo "  docker-dev   Run dev mode with Docker Compose"
	@echo "  docker-stop  Stop Docker containers"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        Clean build artifacts"

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Testing
test:
	python -m pytest tests/ --cov=wenah --cov-report=term-missing --cov-report=html

test-quick:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/ -v --ignore=tests/test_integration.py

test-integration:
	python -m pytest tests/test_integration.py -v

# Code Quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Running
run:
	uvicorn wenah.api.main:app --host 0.0.0.0 --port 8000

run-dev:
	uvicorn wenah.api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t wenah-api:latest .

docker-run:
	docker compose up wenah-api

docker-dev:
	docker compose --profile dev up wenah-dev

docker-stop:
	docker compose down

docker-logs:
	docker compose logs -f

# Maintenance
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Data Management
init-db:
	python scripts/build_embeddings.py

ingest-laws:
	python scripts/ingest_laws.py

validate-rules:
	python scripts/validate_rules.py
