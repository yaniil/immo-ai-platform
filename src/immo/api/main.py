"""API REST (FastAPI) : estimation, indicateurs de marché et assistant conversationnel.

Lancement : uvicorn immo.api.main:app --reload   (documentation interactive sur /docs)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from immo import __version__
from immo.agent.data import CommuneNotFoundError, DataContext, DataNotReadyError
from immo.agent.graph import ImmoAgent
from immo.agent.llm import LLMNotConfiguredError, is_configured
from immo.config import Settings, get_settings
from immo.ml.predict import ModelNotTrainedError

logger = logging.getLogger(__name__)

TypeLocal = Literal["Appartement", "Maison"]


class EstimationRequest(BaseModel):
    commune: str = Field(examples=["Clichy"], description="Nom, code INSEE ou code postal")
    type_local: TypeLocal = "Appartement"
    surface: float = Field(ge=9, le=1000, examples=[45], description="Surface habitable en m²")
    nombre_pieces: int = Field(ge=1, le=20, examples=[2])
    surface_terrain: float = Field(default=0, ge=0, le=100_000, description="Maisons uniquement (m²)")
    nb_dependances: int = Field(default=0, ge=0, le=10, description="Caves ou parkings inclus")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["Quel est le prix médian à Clichy ?"])
    historique: list[Message] = Field(default_factory=list, max_length=20)


class Services:
    """Charge paresseusement les composants : l'API démarre même si le pipeline n'a pas encore tourné."""

    def __init__(self, settings: Settings, agent_factory: Callable[[DataContext], ImmoAgent] | None = None):
        self.settings = settings
        self._agent_factory = agent_factory
        self._ctx: DataContext | None = None
        self._agent: ImmoAgent | None = None

    @property
    def ctx(self) -> DataContext:
        if self._ctx is None:
            try:
                self._ctx = DataContext(self.settings.warehouse_path, self.settings.model_dir)
            except DataNotReadyError as exc:
                raise HTTPException(503, str(exc)) from exc
        return self._ctx

    @property
    def agent(self) -> ImmoAgent:
        if self._agent is None:
            try:
                if self._agent_factory is not None:
                    self._agent = self._agent_factory(self.ctx)
                else:
                    from immo.agent.factory import build_agent

                    self._agent = build_agent(self.settings, ctx=self.ctx)
            except LLMNotConfiguredError as exc:
                raise HTTPException(503, str(exc)) from exc
        return self._agent


def create_app(
    settings: Settings | None = None,
    agent_factory: Callable[[DataContext], ImmoAgent] | None = None,
) -> FastAPI:
    services = Services(settings or get_settings(), agent_factory)
    app = FastAPI(
        title="Immo AI Platform",
        version=__version__,
        description="Prix de l'immobilier en Île-de-France (données DVF) : indicateurs, estimation et assistant IA.",
    )

    def get_services() -> Services:
        return services

    @app.get("/health", tags=["système"])
    def health(s: Services = Depends(get_services)) -> dict:
        etat = {"statut": "ok", "version": __version__,
                "llm": {"fournisseur": s.settings.llm_provider, "modele": s.settings.llm_model,
                        "configure": is_configured(s.settings)}}
        try:
            etat["donnees"] = s.ctx.metadata
        except HTTPException as exc:
            return {**etat, "statut": "donnees_absentes", "detail": exc.detail}
        try:
            meta = s.ctx.model.metadata
            etat["modele_prix"] = {"entraine_le": meta["entraine_le"], "mdape_test": meta["metriques"]["mdape"],
                                   "couverture_intervalle_80": meta["metriques"]["couverture_intervalle_80"]}
        except ModelNotTrainedError:
            etat["modele_prix"] = None
        return etat

    @app.get("/communes", tags=["marché"])
    def communes(recherche: str = "", limite: int = Query(20, ge=1, le=2000),
                 s: Services = Depends(get_services)) -> list[dict]:
        return [c.as_dict() for c in s.ctx.search_communes(recherche, limite)]

    @app.get("/communes/{commune}/tendance", tags=["marché"])
    def tendance(commune: str, type_local: TypeLocal = "Appartement",
                 s: Services = Depends(get_services)) -> dict:
        try:
            c = s.ctx.resolve_commune(commune)
        except CommuneNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"commune": c.as_dict(), "type_local": type_local, "par_annee": s.ctx.trend(c.code_commune, type_local)}

    @app.post("/estimation", tags=["estimation"])
    def estimation(req: EstimationRequest, s: Services = Depends(get_services)) -> dict:
        try:
            c = s.ctx.resolve_commune(req.commune)
            return s.ctx.estimate(c, req.type_local, req.surface, req.nombre_pieces,
                                  req.surface_terrain, req.nb_dependances)
        except CommuneNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ModelNotTrainedError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/assistant", tags=["assistant"])
    async def assistant(req: AssistantRequest, s: Services = Depends(get_services)) -> dict:
        agent = s.agent
        historique = [m.model_dump() for m in req.historique]
        try:
            reponse = await run_in_threadpool(agent.ask, req.question, historique)
        except Exception as exc:  # erreur du fournisseur LLM (quota, réseau...)
            logger.exception("Erreur de l'agent")
            raise HTTPException(502, f"Erreur du modèle de langage : {exc}") from exc
        return reponse.as_dict()

    return app


app = create_app()
