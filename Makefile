.PHONY: all format lint test clean build publish

all: format lint test

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	mypy langchain_kalibr/

test:
	pytest tests/ -v

test-integration:
	pytest tests/ -v -m integration

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

build: clean
	python -m build

publish: build
	twine upload dist/*

publish-test: build
	twine upload --repository testpypi dist/*
