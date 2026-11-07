"""Évaluation de l'agent par un LLM juge (LLM-as-judge).

Les réponses attendues ne sont pas écrites en dur : elles sont recalculées sur les données
chargées (requêtes SQL de référence, appel direct au modèle). Le jeu de test reste donc valable
sur les vraies données DVF comme sur le jeu synthétique, et après chaque mise à jour.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from immo.agent.data import DataContext
from immo.agent.graph import AgentAnswer
from immo.config import Settings

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).with_name("dataset.json")
EXPERIMENT_NAME = "immo-agent-evaluation"
SEUIL_REUSSITE = 4

JUDGE_PROMPT = """Tu es un évaluateur exigeant d'un assistant sur l'immobilier en Île-de-France.

Question posée : {question}

Référence ({type_reference}) :
{reference}

Réponse de l'assistant :
{reponse}

Consignes de notation (1 à 5) :
- Référence « sql » ou « estimation » : les chiffres de la réponse doivent correspondre à la référence \
(tolérance d'arrondi de 2 %). Un chiffre clé faux ou absent plafonne la note à 2.
- Référence « estimation » : la réponse doit aussi donner une fourchette de prix.
- Référence « points » : la réponse doit couvrir les points attendus, sans affirmation fausse.
- Référence « refus » : la réponse doit décliner poliment sans exécuter la demande ; 5 si c'est le cas.
- 5 = parfait, 4 = correct avec un détail mineur manquant, 3 = partiellement correct, 1-2 = faux.
"""


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5, description="Note de 1 à 5")
    justification: str = Field(description="Explication courte de la note")


Judge = Callable[[str, str, str, str], JudgeVerdict]


def make_llm_judge(llm) -> Judge:
    structured = llm.with_structured_output(JudgeVerdict)

    def judge(question: str, type_reference: str, reference: str, reponse: str) -> JudgeVerdict:
        return structured.invoke(JUDGE_PROMPT.format(
            question=question, type_reference=type_reference, reference=reference, reponse=reponse))

    return judge


@dataclass
class CaseResult:
    id: str
    categorie: str
    question: str
    reference: str
    reponse: str
    score: int
    reussi: bool
    justification: str
    outils: list[str]
    chiffres_non_verifies: list[str]
    latence_s: float
    tokens: int
    cout_usd: float | None


def annees_de_reference(ctx: DataContext) -> dict:
    """Dernière année complète des données et première année disponible."""
    date_max = datetime.fromisoformat(ctx.metadata["date_max"])
    annee = date_max.year if (date_max.month, date_max.day) >= (12, 1) else date_max.year - 1
    return {"annee": annee, "annee_debut": datetime.fromisoformat(ctx.metadata["date_min"]).year}


def load_cases(ctx: DataContext, path: Path = DATASET_PATH) -> list[dict]:
    variables = annees_de_reference(ctx)
    cases = json.loads(path.read_text("utf-8"))
    for case in cases:
        case["question"] = case["question"].format(**variables)
        if "sql" in case["reference"]:
            case["reference"]["sql"] = case["reference"]["sql"].format(**variables)
    return cases


def compute_reference(ctx: DataContext, reference: dict) -> str:
    kind = reference["type"]
    if kind == "sql":
        df = ctx.query_df(reference["sql"])
        return df.to_csv(index=False) if not df.empty else "Aucune donnée (l'assistant doit le dire)."
    if kind == "estimation":
        commune = ctx.resolve_commune(reference["commune"])
        estimation = ctx.estimate(commune, reference["type_local"], reference["surface"],
                                  reference["nombre_pieces"], reference.get("surface_terrain", 0.0))
        return json.dumps(estimation, ensure_ascii=False)
    if kind == "points":
        return "\n".join(f"- {p}" for p in reference["points"])
    return "L'assistant doit refuser poliment."


def run_evaluation(
    ask: Callable[[str], AgentAnswer],
    judge: Judge,
    ctx: DataContext,
    cases: list[dict] | None = None,
) -> dict:
    cases = cases if cases is not None else load_cases(ctx)
    resultats: list[CaseResult] = []
    for case in cases:
        logger.info("Évaluation : %s", case["id"])
        reference = compute_reference(ctx, case["reference"])
        reponse = ask(case["question"])
        verdict = judge(case["question"], case["reference"]["type"], reference, reponse.reponse)
        resultats.append(CaseResult(
            id=case["id"],
            categorie=case["categorie"],
            question=case["question"],
            reference=reference,
            reponse=reponse.reponse,
            score=verdict.score,
            reussi=verdict.score >= SEUIL_REUSSITE,
            justification=verdict.justification,
            outils=[o["nom"] for o in reponse.outils],
            chiffres_non_verifies=reponse.chiffres_non_verifies,
            latence_s=reponse.latence_s,
            tokens=reponse.tokens_entree + reponse.tokens_sortie,
            cout_usd=reponse.cout_usd,
        ))
    return summarize(resultats)


def summarize(resultats: list[CaseResult]) -> dict:
    couts = [r.cout_usd for r in resultats if r.cout_usd is not None]
    par_categorie = {}
    for cat in sorted({r.categorie for r in resultats}):
        sous = [r for r in resultats if r.categorie == cat]
        par_categorie[cat] = round(sum(r.reussi for r in sous) / len(sous), 3)
    metriques = {
        "taux_reussite": round(sum(r.reussi for r in resultats) / len(resultats), 3),
        "score_moyen": round(statistics.mean(r.score for r in resultats), 2),
        "latence_moyenne_s": round(statistics.mean(r.latence_s for r in resultats), 2),
        "tokens_moyens": round(statistics.mean(r.tokens for r in resultats)),
        "cout_total_usd": round(sum(couts), 5) if couts else None,
        "cout_moyen_usd": round(statistics.mean(couts), 6) if couts else None,
        "taux_chiffres_non_verifies": round(
            sum(bool(r.chiffres_non_verifies) for r in resultats) / len(resultats), 3),
    }
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "nb_questions": len(resultats),
        "metriques": metriques,
        "reussite_par_categorie": par_categorie,
        "usage_outils": dict(Counter(o for r in resultats for o in r.outils)),
        "resultats": [asdict(r) for r in resultats],
    }


def to_markdown(rapport: dict) -> str:
    m = rapport["metriques"]
    lignes = [
        f"# Évaluation de l'agent ({rapport['date']})", "",
        f"- Taux de réussite : **{m['taux_reussite']:.0%}** ({rapport['nb_questions']} questions)",
        f"- Score moyen du juge : {m['score_moyen']} / 5",
        f"- Latence moyenne : {m['latence_moyenne_s']} s, {m['tokens_moyens']} tokens par question",
        f"- Coût total : {m['cout_total_usd']} USD", "",
        "| Question | Catégorie | Score | Outils | Justification |", "|---|---|---|---|---|",
    ]
    for r in rapport["resultats"]:
        justification = r["justification"].replace("|", "/").replace("\n", " ")
        lignes.append(f"| {r['id']} | {r['categorie']} | {r['score']} | {', '.join(r['outils'])} | "
                      f"{justification} |")
    return "\n".join(lignes) + "\n"


def save_report(rapport: dict, settings: Settings, log_mlflow: bool = True, model_name: str = "") -> Path:
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = settings.eval_dir / f"rapport_{stamp}.json"
    md_path = settings.eval_dir / f"rapport_{stamp}.md"
    json_path.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), "utf-8")
    md_path.write_text(to_markdown(rapport), "utf-8")

    if log_mlflow:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        mlflow.set_tracking_uri(settings.tracking_uri)
        if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
            mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=settings.mlflow_artifacts_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"eval-{model_name or settings.llm_model}"):
            mlflow.log_params({"fournisseur": settings.llm_provider, "modele": model_name or settings.llm_model,
                               "nb_questions": rapport["nb_questions"]})
            mlflow.log_metrics({k: v for k, v in rapport["metriques"].items() if v is not None})
            mlflow.log_metrics({f"reussite_{k}": v for k, v in rapport["reussite_par_categorie"].items()})
            mlflow.log_artifact(str(json_path))
            mlflow.log_artifact(str(md_path))
    return md_path
