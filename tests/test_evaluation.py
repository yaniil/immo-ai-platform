import mlflow

from immo.agent.graph import AgentAnswer
from immo.evaluation.judge import JudgeVerdict, compute_reference, load_cases, run_evaluation, save_report


def test_les_questions_sont_datees_sur_les_donnees(ctx):
    cases = load_cases(ctx)
    assert len(cases) >= 12
    prix = next(c for c in cases if c["id"] == "prix_commune")
    assert "2025" in prix["question"] and "annee = 2025" in prix["reference"]["sql"]


def test_references_calculees_sur_les_donnees(ctx):
    cases = {c["id"]: c for c in load_cases(ctx)}
    assert "prix_m2_median" in compute_reference(ctx, cases["prix_commune"]["reference"])
    assert "prix_estime" in compute_reference(ctx, cases["estimation_appartement"]["reference"])
    assert compute_reference(ctx, cases["hors_sujet"]["reference"]).startswith("L'assistant doit refuser")


def test_harnais_d_evaluation(ctx, settings):
    def faux_agent(question):
        return AgentAnswer(reponse=f"Réponse à : {question}", outils=[{"nom": "requete_sql", "arguments": {}}],
                           latence_s=0.5, tokens_entree=900, tokens_sortie=100, cout_usd=0.0002)

    def faux_juge(question, type_reference, reference, reponse):
        return JudgeVerdict(score=5 if type_reference != "refus" else 2, justification="test")

    rapport = run_evaluation(faux_agent, faux_juge, ctx)
    m = rapport["metriques"]
    assert m["taux_reussite"] == round(11 / 13, 3)
    assert rapport["reussite_par_categorie"]["garde-fou"] == 0
    assert rapport["usage_outils"] == {"requete_sql": 13}

    chemin = save_report(rapport, settings, log_mlflow=True, model_name="faux-modele")
    assert chemin.exists() and "Taux de réussite" in chemin.read_text("utf-8")
    mlflow.set_tracking_uri(settings.tracking_uri)
    runs = mlflow.search_runs(experiment_names=["immo-agent-evaluation"])
    assert len(runs) == 1 and runs["metrics.taux_reussite"].iloc[0] == m["taux_reussite"]
