"""Configuration centralisée, lue depuis les variables d'environnement et le fichier .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paris, Seine-et-Marne, Yvelines, Essonne, Hauts-de-Seine, Seine-Saint-Denis, Val-de-Marne, Val-d'Oise
IDF_DEPARTEMENTS = ["75", "77", "78", "91", "92", "93", "94", "95"]

Source = Literal["dvf", "sample"]


class Settings(BaseSettings):
    """Tous les réglages peuvent être surchargés par une variable IMMO_<NOM> (ex. IMMO_LLM_MODEL)."""

    model_config = SettingsConfigDict(env_prefix="IMMO_", env_file=".env", extra="ignore")

    # Chemins
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    docs_dir: Path = PROJECT_ROOT / "docs" / "knowledge"
    dbt_dir: Path = PROJECT_ROOT / "dbt"

    # Périmètre des données
    departements: list[str] = IDF_DEPARTEMENTS
    annees: list[int] = [2021, 2022, 2023, 2024, 2025]

    # LLM (fournisseurs supportés : groq, anthropic, openai)
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    judge_model: str | None = None  # modèle juge pour l'évaluation (par défaut : llm_model)
    llm_temperature: float = 0.0
    llm_price_input_per_mtok: float | None = None  # surcharge du prix (USD / million de tokens)
    llm_price_output_per_mtok: float | None = None
    agent_max_steps: int = 8

    # RAG
    rag_dense: bool = True  # recherche sémantique si fastembed est installé
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # MLflow (par défaut : base SQLite locale dans data/)
    mlflow_tracking_uri: str | None = None

    def raw_dir(self, source: Source) -> Path:
        return self.data_dir / "raw" / ("dvf" if source == "dvf" else "sample")

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"

    @property
    def model_dir(self) -> Path:
        return self.models_dir / "price_model"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def tracking_uri(self) -> str:
        return self.mlflow_tracking_uri or f"sqlite:///{(self.data_dir / 'mlflow.db').as_posix()}"

    @property
    def mlflow_artifacts_uri(self) -> str:
        return (self.data_dir / "mlartifacts").resolve().as_uri()


@lru_cache
def get_settings() -> Settings:
    load_dotenv()  # expose aussi les clés API (GROQ_API_KEY...) aux bibliothèques LLM
    return Settings()
