"""Lancement des transformations dbt (bronze → silver → gold) dans l'entrepôt DuckDB."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

from immo.config import Settings, Source

logger = logging.getLogger(__name__)


def detect_source(settings: Settings) -> Source:
    """Utilise les vraies données DVF si elles ont été téléchargées, sinon le jeu synthétique."""
    return "dvf" if any(settings.raw_dir("dvf").glob("*/*.csv.gz")) else "sample"


def run_dbt(settings: Settings, source: Source, command: str = "build") -> None:
    """Exécute `dbt build` : construit les modèles ET lance les tests de données.

    dbt tourne dans un processus séparé : DuckDB n'autorise pas, dans un même processus, deux
    connexions au même fichier avec des réglages différents (dbt écrit, l'agent lit en sécurité).
    """
    raw_dir = settings.raw_dir(source)
    if not any(raw_dir.glob("*/*.csv.gz")):
        raise FileNotFoundError(
            f"Aucun fichier dans {raw_dir}. Lancez d'abord : immo ingest --source {source}"
        )

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["IMMO_DATA_DIR"] = settings.data_dir.resolve().as_posix()
    variables = {
        "source": source,
        "raw_glob": (raw_dir.resolve() / "*" / "*.csv.gz").as_posix(),
        "departements": settings.departements,
    }
    args = [
        command,
        "--project-dir", str(settings.dbt_dir),
        "--profiles-dir", str(settings.dbt_dir),
        "--vars", json.dumps(variables),
    ]
    logger.info("dbt %s (source=%s)", command, source)
    lanceur = ("import sys; from dbt.cli.main import dbtRunner; "
               "sys.exit(0 if dbtRunner().invoke(sys.argv[1:]).success else 1)")
    resultat = subprocess.run([sys.executable, "-c", lanceur, *args], env=os.environ.copy())
    if resultat.returncode != 0:
        raise RuntimeError(f"dbt {command} a échoué : voir les logs ci-dessus.")
