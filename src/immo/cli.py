"""Ligne de commande : immo <commande>. Lancez `immo --help` pour la liste."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from immo.config import get_settings


def _ingest(args, settings) -> None:
    annees = [int(a) for a in args.annees.split(",")] if args.annees else settings.annees
    if args.source == "dvf":
        from immo.ingestion.download import download_dvf

        download_dvf(annees, settings.departements, settings.raw_dir("dvf"), force=args.force)
    else:
        from immo.ingestion.sample import generate_sample

        generate_sample(settings.raw_dir("sample"), annees, mutations_par_an=args.n)


def _transform(args, settings) -> None:
    from immo.transform import detect_source, run_dbt

    run_dbt(settings, args.source or detect_source(settings))


def _train(args, settings) -> None:
    from immo.ml.train import train_model

    meta = train_model(settings, test_months=args.test_months, log_mlflow=not args.no_mlflow)
    print(json.dumps(meta["metriques"], indent=2, ensure_ascii=False))


def _pipeline(args, settings) -> None:
    _ingest(args, settings)
    _transform(args, settings)  # même source que l'ingestion
    _train(args, settings)


def _ask(args, settings) -> None:
    from immo.agent.factory import build_agent

    reponse = build_agent(settings).ask(" ".join(args.question))
    print(reponse.reponse)
    print(f"\n[outils : {', '.join(o['nom'] for o in reponse.outils) or 'aucun'} | "
          f"{reponse.tokens_entree + reponse.tokens_sortie} tokens | coût : {reponse.cout_usd} USD | "
          f"{reponse.latence_s} s]")
    if reponse.chiffres_non_verifies:
        print(f"[attention, chiffres non vérifiés : {', '.join(reponse.chiffres_non_verifies)}]")


def _evaluate(args, settings) -> None:
    from immo.agent.factory import build_agent, build_context
    from immo.agent.llm import make_llm
    from immo.evaluation.judge import load_cases, make_llm_judge, run_evaluation, save_report

    ctx = build_context(settings)
    agent = build_agent(settings, ctx=ctx)
    judge = make_llm_judge(make_llm(settings, settings.judge_model))
    cases = load_cases(ctx)[: args.limit] if args.limit else load_cases(ctx)
    rapport = run_evaluation(agent.ask, judge, ctx, cases)
    chemin = save_report(rapport, settings, log_mlflow=not args.no_mlflow, model_name=settings.llm_model)
    print(json.dumps(rapport["metriques"], indent=2, ensure_ascii=False))
    print(f"Rapport : {chemin}")


def _serve(args, settings) -> None:
    import uvicorn

    uvicorn.run("immo.api.main:app", host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immo", description="Immo AI Platform : données DVF, modèle et agent IA.")
    sub = parser.add_subparsers(dest="commande", required=True)

    def add_ingest_args(p):
        p.add_argument("--source", choices=["dvf", "sample"], default="dvf",
                       help="dvf = vraies données data.gouv.fr ; sample = jeu synthétique hors ligne")
        p.add_argument("--annees", help="ex. 2023,2024,2025 (défaut : configuration)")
        p.add_argument("--force", action="store_true", help="retélécharger les fichiers existants")
        p.add_argument("-n", type=int, default=4000, help="mutations par an (jeu synthétique)")

    add_ingest_args(sub.add_parser("ingest", help="télécharger DVF (ou générer le jeu synthétique)"))
    p = sub.add_parser("transform", help="construire l'entrepôt avec dbt (bronze → silver → gold) et le tester")
    p.add_argument("--source", choices=["dvf", "sample"], help="défaut : dvf si téléchargé, sinon sample")
    for nom, aide in [("train", "entraîner le modèle de prix"),
                      ("pipeline", "ingest + transform + train en une commande")]:
        p = sub.add_parser(nom, help=aide)
        if nom == "pipeline":
            add_ingest_args(p)
        p.add_argument("--test-months", type=int, default=6)
        p.add_argument("--no-mlflow", action="store_true")
    p = sub.add_parser("ask", help="poser une question à l'agent")
    p.add_argument("question", nargs="+")
    p = sub.add_parser("evaluate", help="évaluer l'agent avec un LLM juge")
    p.add_argument("--limit", type=int, help="nombre de questions (défaut : toutes)")
    p.add_argument("--no-mlflow", action="store_true")
    p = sub.add_parser("serve", help="lancer l'API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return parser


COMMANDES = {"ingest": _ingest, "transform": _transform, "train": _train, "pipeline": _pipeline,
             "ask": _ask, "evaluate": _evaluate, "serve": _serve}


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    for bruyant in ("httpx", "urllib3", "alembic", "mlflow"):
        logging.getLogger(bruyant).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    try:
        COMMANDES[args.commande](args, get_settings())
    except Exception as exc:
        logging.getLogger("immo").error("%s", exc, exc_info=os.getenv("IMMO_DEBUG") == "1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
