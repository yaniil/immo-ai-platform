# Image Airflow + projet installé dans un environnement Python séparé,
# pour éviter tout conflit entre les dépendances d'Airflow et celles du projet.
FROM apache/airflow:3.3.2-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN python -m venv /opt/immo-venv \
    && /opt/immo-venv/bin/pip install --no-cache-dir -e "/app[pipeline,llm,rag]" \
    && chown -R airflow:root /app /opt/immo-venv

USER airflow
WORKDIR /app
ENV IMMO_BIN=/opt/immo-venv/bin/immo \
    AIRFLOW__CORE__DAGS_FOLDER=/app/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False
