"""Téléchargement des fichiers DVF géolocalisés (Etalab / DGFiP) publiés sur data.gouv.fr."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DVF_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{departement}.csv.gz"


def download_dvf(
    annees: Iterable[int],
    departements: Iterable[str],
    raw_dir: Path,
    force: bool = False,
    timeout: int = 120,
) -> list[Path]:
    """Télécharge un fichier par (année, département) dans raw_dir/<année>/<département>.csv.gz.

    Les fichiers déjà présents sont ignorés (sauf force=True) et une année absente du serveur
    (404) est simplement sautée : la profondeur d'historique de DVF évolue à chaque publication.
    """
    fichiers: list[Path] = []
    with requests.Session() as session:
        for annee in annees:
            for departement in departements:
                destination = raw_dir / str(annee) / f"{departement}.csv.gz"
                if destination.exists() and not force:
                    logger.info("Déjà présent : %s", destination)
                    fichiers.append(destination)
                    continue

                url = DVF_URL.format(annee=annee, departement=departement)
                logger.info("Téléchargement %s", url)
                reponse = session.get(url, stream=True, timeout=timeout)
                if reponse.status_code == 404:
                    logger.warning("Non publié sur data.gouv.fr : %s %s", annee, departement)
                    continue
                reponse.raise_for_status()

                destination.parent.mkdir(parents=True, exist_ok=True)
                temporaire = destination.with_suffix(".part")
                with temporaire.open("wb") as f:
                    for bloc in reponse.iter_content(chunk_size=1 << 20):
                        f.write(bloc)
                temporaire.replace(destination)  # écriture atomique : pas de fichier à moitié écrit
                fichiers.append(destination)

    if not fichiers:
        raise RuntimeError("Aucun fichier DVF téléchargé : vérifiez les années demandées et la connexion.")
    return fichiers
