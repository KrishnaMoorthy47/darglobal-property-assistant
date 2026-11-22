PY ?= python

.PHONY: dev test eval data build run

dev:
	$(PY) -m uvicorn app.main:app --reload --port 7860

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m evals.run

data:
	$(PY) -m scrapers.darglobal
	$(PY) -m scrapers.wasalt --n 1200
	$(PY) -m pipeline.build --enrich

build:
	docker build -t property-assistant .

run:
	docker run -p 7860:7860 --env-file .env property-assistant
