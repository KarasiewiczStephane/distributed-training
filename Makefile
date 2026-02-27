IMAGE_NAME := $(shell basename $(CURDIR))

.PHONY: install test lint clean run train-ddp train-horovod \
        docker docker-gpu docker-ddp docker-benchmark

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

train-ddp:
	torchrun --nproc_per_node=2 -m src.main --config configs/training.yaml --distributed

train-horovod:
	horovodrun -np 2 python -m src.main --trainer horovod --config configs/training.yaml

docker:
	docker build -t $(IMAGE_NAME) .

docker-gpu:
	docker build -t $(IMAGE_NAME) .
	docker run --gpus all $(IMAGE_NAME)

docker-ddp:
	docker compose up trainer-ddp

docker-benchmark:
	docker compose up benchmark
