"""Fixtures partagées : un entrepôt complet est construit une fois pour toute la session de tests
(jeu synthétique → dbt build → entraînement du modèle), dans un dossier temporaire."""

from __future__ import annotations

import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from immo.agent.data import DataContext
from immo.agent.rag import build_retriever
from immo.config import Settings
from immo.ingestion.sample import generate_sample
from immo.ml.train import train_model
from immo.transform import run_dbt

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")


class ScriptedChatModel(GenericFakeChatModel):
    """Faux LLM qui renvoie des messages écrits à l'avance (y compris des appels d'outils)."""

    def bind_tools(self, tools, **kwargs):
        return self


USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}


def ai(content: str = "", tool_calls: list[dict] | None = None) -> AIMessage:
    calls = [{"name": c["name"], "args": c["args"], "id": f"appel_{i}"} for i, c in enumerate(tool_calls or [])]
    return AIMessage(content=content, tool_calls=calls, usage_metadata=USAGE)


def scripted(*messages: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("immo")
    s = Settings(
        data_dir=root / "data",
        models_dir=root / "models",
        annees=[2024, 2025],
        rag_dense=False,
        mlflow_tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        llm_provider="groq",
        llm_model="openai/gpt-oss-120b",
        llm_price_input_per_mtok=None,
        llm_price_output_per_mtok=None,
    )
    generate_sample(s.raw_dir("sample"), s.annees, mutations_par_an=2500, seed=7)
    run_dbt(s, "sample")
    return s


@pytest.fixture(scope="session")
def trained(settings) -> dict:
    return train_model(settings, log_mlflow=True)


@pytest.fixture(scope="session")
def ctx(settings, trained) -> DataContext:
    return DataContext(settings.warehouse_path, settings.model_dir)


@pytest.fixture(scope="session")
def retriever(settings):
    return build_retriever(settings.docs_dir, dense=False)
