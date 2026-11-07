"""Entraînement du modèle d'estimation du prix au m² (LightGBM), suivi dans MLflow.

Choix principaux :
- cible = log(prix au m²) : l'erreur devient relative (10 % d'erreur pèse autant à Paris qu'à Meaux) ;
- validation temporelle : on teste sur les 6 derniers mois, comme en production où l'on estime
  des ventes futures (un découpage aléatoire serait trop optimiste) ;
- deux modèles quantiles (10 % et 90 %) donnent une fourchette de prix à 80 % ;
- comparaison à une référence simple : la médiane de la commune pour le même type de bien.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from immo.config import Settings
from immo.ml.features import CATEGORICAL_FEATURES, FEATURES, TARGET, build_categories, prepare_features

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "immo-prix-m2"
REGISTERED_MODEL_NAME = "immo-prix-m2"
QUANTILES = {"median": None, "q10": 0.1, "q90": 0.9}

BASE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 10,
    "seed": 42,
    "verbosity": -1,
}


def _params(name: str, base: dict) -> dict:
    alpha = QUANTILES[name]
    if alpha is None:
        return {**base, "objective": "regression"}
    return {**base, "objective": "quantile", "alpha": alpha}


def load_training_data(warehouse_path: Path) -> tuple[pd.DataFrame, dict]:
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        df = con.execute("select * from gold.ml_features").df()
        metadata = con.execute("select * from gold.metadata").df().iloc[0].to_dict()
    df["date_mutation"] = pd.to_datetime(df["date_mutation"])
    return df, metadata


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Référence naïve : médiane commune × type, puis département × type, puis globale."""
    pred = pd.Series(np.nan, index=test.index)
    for keys in (["code_commune", "type_local"], ["code_departement", "type_local"]):
        medianes = train.groupby(keys, observed=True)[TARGET].median().rename("ref").reset_index()
        merged = test[keys].merge(medianes, on=keys, how="left")
        pred = pred.fillna(pd.Series(merged["ref"].to_numpy(), index=test.index))
    return pred.fillna(train[TARGET].median()).to_numpy()


def compute_metrics(y_true: np.ndarray, pred: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    ape = np.abs(pred - y_true) / y_true
    log_true, log_pred = np.log(y_true), np.log(pred)
    ss_res = np.sum((log_true - log_pred) ** 2)
    ss_tot = np.sum((log_true - log_true.mean()) ** 2)
    return {
        "mae_prix_m2": float(np.mean(np.abs(pred - y_true))),
        "mdape": float(np.median(ape)),
        "mape": float(np.mean(ape)),
        "r2_log": float(1 - ss_res / ss_tot),
        "couverture_intervalle_80": float(np.mean((y_true >= low) & (y_true <= high))),
        "largeur_intervalle_mediane": float(np.median((high - low) / pred)),
    }


def _fit(name: str, params: dict, X: pd.DataFrame, y: np.ndarray, rounds: int,
         X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None) -> lgb.Booster:
    train_set = lgb.Dataset(X, y, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    if X_val is None:
        return lgb.train(_params(name, params), train_set, num_boost_round=rounds)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    return lgb.train(
        _params(name, params), train_set, num_boost_round=rounds, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def train_model(
    settings: Settings,
    test_months: int = 6,
    max_rounds: int = 3000,
    params: dict | None = None,
    log_mlflow: bool = True,
) -> dict:
    """Entraîne, évalue sur les derniers mois, réentraîne sur tout l'historique et sauvegarde."""
    params = {**BASE_PARAMS, **(params or {})}
    df, data_meta = load_training_data(settings.warehouse_path)
    if len(df) < 500:
        raise ValueError(f"Trop peu de ventes pour entraîner un modèle ({len(df)}).")

    date_max = df["date_mutation"].max()
    test_start = date_max - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=3)
    train_all = df[df["date_mutation"] <= test_start]
    test = df[df["date_mutation"] > test_start]
    fit = train_all[train_all["date_mutation"] <= val_start]
    val = train_all[train_all["date_mutation"] > val_start]

    categories = build_categories(df)
    X_fit, X_val, X_test = (prepare_features(d, categories) for d in (fit, val, test))
    y_fit, y_val = np.log(fit[TARGET].to_numpy()), np.log(val[TARGET].to_numpy())

    # 1) Évaluation honnête sur une période jamais vue
    best_rounds, test_preds = {}, {}
    for name in QUANTILES:
        booster = _fit(name, params, X_fit, y_fit, max_rounds, X_val, y_val)
        best_rounds[name] = max(booster.best_iteration or max_rounds, 50)
        test_preds[name] = np.exp(booster.predict(X_test, num_iteration=best_rounds[name]))
        logger.info("Modèle %s : %d itérations", name, best_rounds[name])

    low, mid, high = np.sort(np.vstack([test_preds["q10"], test_preds["median"], test_preds["q90"]]), axis=0)
    y_test = test[TARGET].to_numpy()
    metrics = compute_metrics(y_test, mid, low, high)
    baseline = baseline_predictions(train_all, test)
    metrics["mdape_reference_mediane_commune"] = float(np.median(np.abs(baseline - y_test) / y_test))
    metrics["gain_vs_reference"] = 1 - metrics["mdape"] / metrics["mdape_reference_mediane_commune"]

    # 2) Modèles finaux réentraînés sur tout l'historique, avec le nombre d'itérations trouvé
    X_all = prepare_features(df, categories)
    y_all = np.log(df[TARGET].to_numpy())
    model_dir = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    finals = {}
    for name in QUANTILES:
        finals[name] = _fit(name, params, X_all, y_all, best_rounds[name])
        finals[name].save_model(str(model_dir / f"{name}.txt"))

    importance = pd.DataFrame({
        "variable": FEATURES,
        "importance_gain": finals["median"].feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(model_dir / "importance_variables.csv", index=False)

    metadata = {
        "entraine_le": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_donnees": data_meta["source_donnees"],
        "periode_donnees": [str(df["date_mutation"].min().date()), str(date_max.date())],
        "periode_test": [str(test["date_mutation"].min().date()), str(date_max.date())],
        "periode_max": float(df["periode"].max()),
        "n_entrainement": len(train_all),
        "n_test": len(test),
        "iterations": best_rounds,
        "parametres": params,
        "variables": FEATURES,
        "categories": categories,
        "metriques": metrics,
    }

    if log_mlflow:
        metadata["mlflow_run_id"] = _log_mlflow(settings, metadata, finals["median"], model_dir)

    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), "utf-8")
    logger.info(
        "MdAPE test = %.1f %% (référence médiane commune : %.1f %%), couverture 80 %% = %.0f %%",
        100 * metrics["mdape"], 100 * metrics["mdape_reference_mediane_commune"],
        100 * metrics["couverture_intervalle_80"],
    )
    return metadata


def _log_mlflow(settings: Settings, metadata: dict, booster: lgb.Booster, model_dir: Path) -> str:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    import mlflow.lightgbm

    mlflow.set_tracking_uri(settings.tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=settings.mlflow_artifacts_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"lightgbm-{metadata['source_donnees']}") as run:
        mlflow.set_tags({"source_donnees": metadata["source_donnees"], "modele": "LightGBM"})
        mlflow.log_params({
            **{k: v for k, v in metadata["parametres"].items() if k != "verbosity"},
            **{f"iterations_{k}": v for k, v in metadata["iterations"].items()},
            "n_entrainement": metadata["n_entrainement"],
            "n_test": metadata["n_test"],
            "periode_test": " → ".join(metadata["periode_test"]),
        })
        mlflow.log_metrics(metadata["metriques"])
        mlflow.log_artifacts(str(model_dir), artifact_path="price_model")
        mlflow.lightgbm.log_model(booster, name="model", registered_model_name=REGISTERED_MODEL_NAME)
        return run.info.run_id
