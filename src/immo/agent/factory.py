"""Assemblage des composants (données, RAG, LLM) pour la CLI et l'API."""

from __future__ import annotations

from immo.agent.data import DataContext
from immo.agent.graph import ImmoAgent
from immo.agent.llm import make_llm
from immo.agent.rag import HybridRetriever, build_retriever
from immo.config import Settings


def build_context(settings: Settings) -> DataContext:
    return DataContext(settings.warehouse_path, settings.model_dir)


def build_default_retriever(settings: Settings) -> HybridRetriever:
    return build_retriever(settings.docs_dir, dense=settings.rag_dense, embedding_model=settings.embedding_model)


def build_agent(
    settings: Settings,
    ctx: DataContext | None = None,
    retriever: HybridRetriever | None = None,
) -> ImmoAgent:
    return ImmoAgent(
        make_llm(settings),
        ctx or build_context(settings),
        retriever or build_default_retriever(settings),
        settings,
    )
