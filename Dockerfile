FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_DISABLE_TELEMETRY=true

# libgomp1 : bibliothèque OpenMP nécessaire à LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dépendances d'abord : la couche est réutilisée tant que pyproject.toml ne change pas
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[pipeline,llm,rag,app]"
COPY . .

RUN useradd --create-home immo && mkdir -p data models && chown -R immo /app
USER immo

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "immo.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
