MODEL ?= voting_hard
MSG ?= $(MODEL) @ $(shell git rev-parse --short HEAD)

.PHONY: install lint format test clean data features predict submit wcg

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

data:
	kaggle competitions download -c titanic -p data/raw
	python -m zipfile -e data/raw/titanic.zip data/raw/
	rm data/raw/titanic.zip

features:
	python -m titanic.features

baseline:
	python -m titanic.evaluate

balance:
	python -m titanic.balancing

tune:
	python -m titanic.tuning

ensemble:
	python -m titanic.ensemble

predict:
	python -m titanic.predict

submit:
	kaggle competitions submit -c titanic -f data/submissions/$(MODEL).csv -m "$(MSG)"
	kaggle competitions submissions titanic

wcg:
	python -m titanic.wcg
