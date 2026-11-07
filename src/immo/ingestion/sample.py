"""Jeu de données SYNTHÉTIQUE au format DVF géolocalisé.

Il sert aux tests, à la CI et à une démo hors ligne : ce ne sont PAS de vraies ventes.
Les niveaux de prix sont des ordres de grandeur réalistes, et le générateur reproduit
volontairement les pièges des vraies données (VEFA, ventes de plusieurs lots, locaux
commerciaux, prix aberrants, lignes dupliquées) pour que le nettoyage soit réellement testé.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colonnes du fichier DVF géolocalisé, dans l'ordre de publication
DVF_COLUMNS = [
    "id_mutation", "date_mutation", "numero_disposition", "nature_mutation", "valeur_fonciere",
    "adresse_numero", "adresse_suffixe", "adresse_nom_voie", "adresse_code_voie", "code_postal",
    "code_commune", "nom_commune", "code_departement", "ancien_code_commune", "ancien_nom_commune",
    "id_parcelle", "ancien_id_parcelle", "numero_volume", "lot1_numero", "lot1_surface_carrez",
    "lot2_numero", "lot2_surface_carrez", "lot3_numero", "lot3_surface_carrez", "lot4_numero",
    "lot4_surface_carrez", "lot5_numero", "lot5_surface_carrez", "nombre_lots", "code_type_local",
    "type_local", "surface_reelle_bati", "nombre_pieces_principales", "code_nature_culture",
    "nature_culture", "code_nature_culture_speciale", "nature_culture_speciale", "surface_terrain",
    "longitude", "latitude",
]

TYPE_CODES = {
    "Maison": "1",
    "Appartement": "2",
    "Dépendance": "3",
    "Local industriel. commercial ou assimilé": "4",
}

# Évolution moyenne des prix (base 2021), proche de la tendance francilienne récente
INDICE_ANNUEL = {2020: 0.98, 2021: 1.00, 2022: 1.02, 2023: 0.97, 2024: 0.92, 2025: 0.93, 2026: 0.94}


@dataclass(frozen=True)
class CommuneRef:
    code_commune: str
    nom_commune: str
    code_postal: str
    latitude: float
    longitude: float
    prix_m2: float  # niveau de prix d'un appartement en 2021 (ordre de grandeur)
    part_maisons: float
    poids: float  # volume relatif de ventes


COMMUNES = [
    CommuneRef("75106", "Paris 6e Arrondissement", "75006", 48.849, 2.333, 14500, 0.00, 1.0),
    CommuneRef("75108", "Paris 8e Arrondissement", "75008", 48.873, 2.312, 12500, 0.00, 1.0),
    CommuneRef("75111", "Paris 11e Arrondissement", "75011", 48.859, 2.380, 10200, 0.00, 3.0),
    CommuneRef("75115", "Paris 15e Arrondissement", "75015", 48.840, 2.293, 10300, 0.00, 4.0),
    CommuneRef("75117", "Paris 17e Arrondissement", "75017", 48.887, 2.307, 10700, 0.00, 3.0),
    CommuneRef("75119", "Paris 19e Arrondissement", "75019", 48.887, 2.385, 8600, 0.00, 2.5),
    CommuneRef("77186", "Fontainebleau", "77300", 48.404, 2.701, 4800, 0.45, 0.6),
    CommuneRef("77284", "Meaux", "77100", 48.960, 2.879, 2900, 0.40, 1.0),
    CommuneRef("78551", "Saint-Germain-en-Laye", "78100", 48.898, 2.093, 7000, 0.35, 1.2),
    CommuneRef("78646", "Versailles", "78000", 48.805, 2.130, 7600, 0.25, 1.6),
    CommuneRef("91228", "Évry-Courcouronnes", "91000", 48.629, 2.441, 2500, 0.35, 0.9),
    CommuneRef("91377", "Massy", "91300", 48.730, 2.271, 4800, 0.25, 1.0),
    CommuneRef("92004", "Asnières-sur-Seine", "92600", 48.914, 2.285, 6800, 0.12, 1.5),
    CommuneRef("92012", "Boulogne-Billancourt", "92100", 48.835, 2.240, 9000, 0.06, 2.0),
    CommuneRef("92024", "Clichy", "92110", 48.904, 2.306, 7300, 0.03, 1.2),
    CommuneRef("92044", "Levallois-Perret", "92300", 48.895, 2.287, 9500, 0.02, 1.0),
    CommuneRef("92050", "Nanterre", "92000", 48.892, 2.206, 5600, 0.15, 1.2),
    CommuneRef("92051", "Neuilly-sur-Seine", "92200", 48.885, 2.268, 11000, 0.05, 1.0),
    CommuneRef("93001", "Aubervilliers", "93300", 48.914, 2.382, 4200, 0.08, 0.9),
    CommuneRef("93048", "Montreuil", "93100", 48.861, 2.443, 6400, 0.20, 1.5),
    CommuneRef("93066", "Saint-Denis", "93200", 48.936, 2.357, 4300, 0.08, 1.3),
    CommuneRef("94028", "Créteil", "94000", 48.790, 2.455, 4200, 0.15, 1.0),
    CommuneRef("94080", "Vincennes", "94300", 48.847, 2.439, 9000, 0.05, 1.0),
    CommuneRef("94081", "Vitry-sur-Seine", "94400", 48.787, 2.392, 4300, 0.25, 1.0),
    CommuneRef("95018", "Argenteuil", "95100", 48.947, 2.248, 3400, 0.40, 1.2),
    CommuneRef("95127", "Cergy", "95000", 49.036, 2.076, 3300, 0.35, 1.0),
]


def _ligne(base: dict, **valeurs) -> dict:
    ligne = dict(base)
    ligne.update(valeurs)
    if "type_local" in valeurs:
        ligne["code_type_local"] = TYPE_CODES.get(valeurs["type_local"], "")
    return ligne


def _logement(rng: np.random.Generator, commune: CommuneRef, type_local: str, annee: int, mois: int):
    """Tire un logement et son prix de vente total."""
    if type_local == "Appartement":
        surface = float(np.clip(rng.lognormal(np.log(52 if commune.part_maisons else 45), 0.45), 12, 250))
        pieces = int(np.clip(round(surface / 22 + rng.normal(0.4, 0.6)), 1, 8))
        facteur_type = 1.0
    else:
        surface = float(np.clip(rng.lognormal(np.log(105), 0.35), 40, 400))
        pieces = int(np.clip(round(surface / 24 + rng.normal(0.5, 0.8)), 2, 12))
        facteur_type = 0.92
    indice = INDICE_ANNUEL.get(annee, 0.94) * (1 + 0.003 * (mois - 6.5))
    effet_surface = (surface / 50) ** -0.08  # les petites surfaces sont plus chères au m²
    prix_m2 = commune.prix_m2 * facteur_type * indice * effet_surface * rng.lognormal(0, 0.12)
    return round(surface), pieces, prix_m2 * round(surface)


def generate_sample(
    raw_dir: Path,
    annees: list[int],
    mutations_par_an: int = 4000,
    seed: int = 42,
) -> list[Path]:
    """Écrit un fichier raw_dir/<année>/<département>.csv.gz par année et département."""
    rng = np.random.default_rng(seed)
    poids = np.array([c.poids for c in COMMUNES])
    poids = poids / poids.sum()
    lignes: list[dict] = []

    for annee in annees:
        for numero in range(1, mutations_par_an + 1):
            commune = COMMUNES[rng.choice(len(COMMUNES), p=poids)]
            date = pd.Timestamp(annee, 1, 1) + pd.Timedelta(days=int(rng.integers(0, 365)))
            date = min(date, pd.Timestamp(annee, 12, 31))
            lat = commune.latitude + rng.normal(0, 0.006)
            lon = commune.longitude + rng.normal(0, 0.008)
            base = {col: "" for col in DVF_COLUMNS}
            base.update(
                id_mutation=f"{annee}-{numero}",
                date_mutation=date.strftime("%Y-%m-%d"),
                numero_disposition="000001",
                nature_mutation="Vente",
                code_postal=commune.code_postal,
                code_commune=commune.code_commune,
                nom_commune=commune.nom_commune,
                code_departement=commune.code_commune[:2],
                id_parcelle=f"{commune.code_commune}000AB{numero % 9999:04d}",
                longitude=f"{lon:.6f}",
                latitude=f"{lat:.6f}",
            )
            type_local = "Maison" if rng.random() < commune.part_maisons else "Appartement"
            surface, pieces, prix = _logement(rng, commune, type_local, annee, date.month)
            terrain = round(rng.lognormal(np.log(350), 0.6)) if type_local == "Maison" else ""
            u = rng.random()

            if u < 0.03:  # vente en l'état futur d'achèvement (neuf)
                base["nature_mutation"] = "Vente en l'état futur d'achèvement"
                prix *= 1.2
            elif u < 0.04:
                base["nature_mutation"] = "Echange"
            elif u < 0.05:  # terrain seul : aucun local
                base["nature_mutation"] = "Vente terrain à bâtir"
                base["valeur_fonciere"] = f"{round(rng.uniform(80_000, 400_000), -2):.2f}"
                lignes.append(_ligne(base, surface_terrain=str(round(rng.uniform(300, 1500)))))
                continue
            elif u < 0.055:  # prix aberrant (vente à l'euro symbolique ou erreur de saisie)
                prix = 1.0 if rng.random() < 0.5 else prix * 25

            base["valeur_fonciere"] = f"{round(prix, -2):.2f}"
            logement = dict(
                type_local=type_local,
                surface_reelle_bati=str(surface),
                nombre_pieces_principales=str(pieces),
                surface_terrain=str(terrain),
            )
            if type_local == "Appartement":
                logement.update(lot1_numero=str(rng.integers(1, 300)), lot1_surface_carrez=f"{surface * 0.97:.2f}")
            lignes.append(_ligne(base, **logement))

            if rng.random() < 0.02:  # même local répété sur une 2e parcelle (doublon fréquent dans DVF)
                lignes.append(_ligne(base, **logement))
            if 0.06 <= u < 0.08:  # vente de plusieurs logements pour un prix global
                surface2, pieces2, _ = _logement(rng, commune, "Appartement", annee, date.month)
                lignes.append(_ligne(base, type_local="Appartement", surface_reelle_bati=str(surface2),
                                     nombre_pieces_principales=str(pieces2)))
            elif 0.08 <= u < 0.09:  # logement + local commercial
                lignes.append(_ligne(base, type_local="Local industriel. commercial ou assimilé",
                                     surface_reelle_bati=str(round(rng.uniform(20, 200))),
                                     nombre_pieces_principales="0"))
            if rng.random() < 0.3:  # cave ou parking vendu avec le logement
                lignes.append(_ligne(base, type_local="Dépendance", nombre_pieces_principales="0"))

    df = pd.DataFrame(lignes, columns=DVF_COLUMNS)
    df["_annee"] = df["date_mutation"].str[:4]
    fichiers = []
    for (annee, departement), groupe in df.groupby(["_annee", "code_departement"]):
        destination = raw_dir / annee / f"{departement}.csv.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        groupe.drop(columns="_annee").to_csv(destination, index=False, compression="gzip")
        fichiers.append(destination)
    (raw_dir / "LISEZMOI.txt").write_text(
        "Données SYNTHÉTIQUES générées par `immo ingest --source sample` : ce ne sont pas de vraies ventes.\n",
        encoding="utf-8",
    )
    logger.info("%d lignes synthétiques écrites dans %d fichiers", len(df), len(fichiers))
    return fichiers
