"""Variables du modèle de prix et préparation des données (partagées entre entraînement et prédiction)."""

from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces",
    "surface_terrain",
    "nb_dependances",
    "longitude",
    "latitude",
    "periode",  # année + mois/12 : capte l'évolution du marché
]
CATEGORICAL_FEATURES = ["type_local", "code_departement", "code_commune"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "prix_m2"


def build_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """Liste figée des modalités : l'entraînement et la prédiction doivent utiliser les mêmes."""
    return {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in CATEGORICAL_FEATURES}


def prepare_features(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Sélectionne les variables et type les catégories (une modalité inconnue devient NaN)."""
    X = df[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    for col in CATEGORICAL_FEATURES:
        X[col] = pd.Categorical(X[col].astype("string"), categories=categories[col])
    return X
