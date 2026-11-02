MODEL ?= voting_hard
MSG ?= $(MODEL) @ $(shell git rev-parse --short HEAD)
IMAGE ?= titanic-duelo
DOCKER_RUN = docker run --rm \
	-v $(CURDIR)/data:/app/data \
	-v $(CURDIR)/reports:/app/reports \
	-v $(CURDIR)/models:/app/models \
	-v $(HOME)/.kaggle:/home/app/.kaggle:ro \
	$(IMAGE)

.PHONY: install lint format test clean data features predict submit wcg docker-build

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

docker-build:
	docker build --build-arg UID=$(shell id -u) -t $(IMAGE) .

docker-%:
	$(DOCKER_RUN) $*
