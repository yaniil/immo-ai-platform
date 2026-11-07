"""DAG Airflow : ingestion DVF → dbt (bronze/silver/gold + tests) → modèle de prix → évaluation de l'agent.

La logique métier vit dans le paquet `immo` : chaque tâche appelle simplement la CLI,
ce qui permet de lancer exactement les mêmes étapes en local, en CI ou dans Airflow.
DVF est mis à jour deux fois par an : une exécution mensuelle suffit (les fichiers déjà
téléchargés ne sont pas retéléchargés).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, task

IMMO = os.environ.get("IMMO_BIN", "immo")  # chemin de la CLI (environnement Python dédié)


@dag(
    dag_id="immo_dvf_pipeline",
    description="Pipeline DVF : ingestion, lakehouse dbt, modèle LightGBM, évaluation de l'agent",
    schedule="0 6 1 * *",  # le 1er de chaque mois à 6 h
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    params={"source": Param("dvf", enum=["dvf", "sample"], description="dvf = vraies données, sample = démo")},
    tags=["immo", "dvf", "ml"],
)
def immo_dvf_pipeline():
    ingestion = BashOperator(
        task_id="ingestion",
        bash_command=f"{IMMO} ingest --source {{{{ params.source }}}}",
        execution_timeout=timedelta(hours=1),
    )
    transformation = BashOperator(
        task_id="dbt_build",
        bash_command=f"{IMMO} transform --source {{{{ params.source }}}}",
    )
    entrainement = BashOperator(task_id="entrainement_modele", bash_command=f"{IMMO} train")

    @task.short_circuit(task_id="llm_configure")
    def llm_configure() -> bool:
        """L'évaluation de l'agent n'a lieu que si une clé API LLM est disponible."""
        return any(os.getenv(k) for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"))

    evaluation = BashOperator(task_id="evaluation_agent", bash_command=f"{IMMO} evaluate")

    ingestion >> transformation >> entrainement >> llm_configure() >> evaluation


immo_dvf_pipeline()
