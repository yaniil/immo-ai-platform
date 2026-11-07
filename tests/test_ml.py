import mlflow
import pytest

from immo.ml.predict import ModelNotTrainedError, PriceModel


def test_metriques_du_modele(trained):
    m = trained["metriques"]
    assert m["mdape"] < 0.2
    # Sur ce petit jeu synthétique, on vérifie seulement que le modèle n'est pas pire que la référence
    # (sur les vraies données, l'écart mesuré est enregistré dans MLflow).
    assert m["mdape"] <= 1.1 * m["mdape_reference_mediane_commune"]
    assert 0.5 <= m["couverture_intervalle_80"] <= 0.95
    assert trained["n_test"] > 0 and trained["periode_test"][1] == "2025-12-31"


def test_le_run_est_enregistre_dans_mlflow(settings, trained):
    mlflow.set_tracking_uri(settings.tracking_uri)
    run = mlflow.get_run(trained["mlflow_run_id"])
    assert run.data.metrics["mdape"] == pytest.approx(trained["metriques"]["mdape"])
    assert run.data.tags["source_donnees"] == "sample"


def test_estimation_coherente(settings, trained):
    model = PriceModel(settings.model_dir)
    commun = dict(code_commune="92024", code_departement="92", longitude=2.306, latitude=48.904)
    petit = model.estimate(type_local="Appartement", surface=30, nombre_pieces=1, **commun)
    grand = model.estimate(type_local="Appartement", surface=90, nombre_pieces=4, **commun)
    for e in (petit, grand):
        assert e["fourchette_basse"] <= e["prix_estime"] <= e["fourchette_haute"]
    assert grand["prix_estime"] > petit["prix_estime"]
    assert petit["commune_connue_du_modele"]


def test_commune_inconnue_signalee(settings, trained):
    e = PriceModel(settings.model_dir).estimate(
        type_local="Appartement", surface=50, nombre_pieces=2, code_commune="99999",
        code_departement="92", longitude=2.3, latitude=48.9)
    assert not e["commune_connue_du_modele"]


def test_modele_absent(tmp_path):
    with pytest.raises(ModelNotTrainedError):
        PriceModel(tmp_path)
