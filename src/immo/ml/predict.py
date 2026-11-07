"""Chargement du modèle de prix et estimation d'un bien."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from immo.ml.features import prepare_features


class ModelNotTrainedError(FileNotFoundError):
    pass


class PriceModel:
    """Trois boosters LightGBM : médiane, quantile 10 % et quantile 90 % du log(prix au m²)."""

    def __init__(self, model_dir: Path):
        if not (model_dir / "metadata.json").exists():
            raise ModelNotTrainedError(f"Aucun modèle dans {model_dir}. Lancez : immo train")
        self.metadata = json.loads((model_dir / "metadata.json").read_text("utf-8"))
        self.categories = self.metadata["categories"]
        self.boosters = {
            name: lgb.Booster(model_file=str(model_dir / f"{name}.txt")) for name in ("median", "q10", "q90")
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie prix_m2_bas, prix_m2 et prix_m2_haut pour chaque ligne."""
        X = prepare_features(df, self.categories)
        preds = np.vstack([np.exp(self.boosters[n].predict(X)) for n in ("q10", "median", "q90")])
        low, mid, high = np.sort(preds, axis=0)  # garantit bas <= estimation <= haut
        return pd.DataFrame({"prix_m2_bas": low, "prix_m2": mid, "prix_m2_haut": high}, index=df.index)

    def estimate(
        self,
        *,
        type_local: str,
        surface: float,
        nombre_pieces: int,
        code_commune: str,
        code_departement: str,
        longitude: float | None,
        latitude: float | None,
        surface_terrain: float = 0.0,
        nb_dependances: int = 0,
    ) -> dict:
        """Estime un bien vendu à la date la plus récente des données."""
        row = pd.DataFrame([{
            "type_local": type_local,
            "surface_reelle_bati": surface,
            "nombre_pieces": nombre_pieces,
            "surface_terrain": surface_terrain if type_local == "Maison" else 0.0,
            "nb_dependances": nb_dependances,
            "code_commune": code_commune,
            "code_departement": code_departement,
            "longitude": longitude,
            "latitude": latitude,
            "periode": self.metadata["periode_max"],
        }])
        p = self.predict(row).iloc[0]
        return {
            "prix_estime": int(round(p["prix_m2"] * surface, -3)),
            "fourchette_basse": int(round(p["prix_m2_bas"] * surface, -3)),
            "fourchette_haute": int(round(p["prix_m2_haut"] * surface, -3)),
            "prix_m2_estime": int(round(p["prix_m2"], -1)),
            "niveau_confiance_fourchette": "80 %",
            "commune_connue_du_modele": code_commune in self.categories["code_commune"],
        }
