# Raccourcis (Linux / macOS / WSL). Sous Windows, utilisez directement les commandes `immo ...`.
.PHONY: install demo data pipeline test lint api app eval mlflow up

install:
	pip install -e ".[pipeline,llm,rag,app,dev]"

demo:            ## pipeline complet sur le jeu synthétique (hors ligne)
	immo pipeline --source sample

data:            ## pipeline complet sur les vraies données DVF
	immo pipeline --source dvf

test:
	pytest -q

lint:
	ruff check .

api:
	uvicorn immo.api.main:app --reload

app:
	streamlit run app/streamlit_app.py

eval:
	immo evaluate

mlflow:
	mlflow ui --backend-store-uri sqlite:///data/mlflow.db

up:
	docker compose up --build
