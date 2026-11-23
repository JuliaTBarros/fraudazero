.PHONY: up down logs reset train check test fmt

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

reset:
	docker compose down -v

train:
	docker compose exec airflow airflow dags trigger fraudazero_pipeline

check:
	docker compose config -q && echo "docker-compose.yml ok"

test:
	python -m pytest -q tests

fmt:
	ruff format .
