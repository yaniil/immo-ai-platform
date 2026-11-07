"""Création du modèle de langage et calcul du coût des appels."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from immo.config import Settings

API_KEY_ENV = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

# Prix indicatifs en USD par million de tokens (entrée, sortie), relevés sur les pages tarifs
# des fournisseurs en 2026 : à vérifier, ou à surcharger via IMMO_LLM_PRICE_INPUT_PER_MTOK / OUTPUT.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3-32b": (0.29, 0.59),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


class LLMNotConfiguredError(RuntimeError):
    pass


def is_configured(settings: Settings) -> bool:
    env = API_KEY_ENV.get(settings.llm_provider)
    return bool(env and os.getenv(env))


def make_llm(settings: Settings, model: str | None = None) -> BaseChatModel:
    provider = settings.llm_provider
    if provider not in API_KEY_ENV:
        raise LLMNotConfiguredError(f"Fournisseur inconnu : {provider} (choix : {', '.join(API_KEY_ENV)}).")
    if not is_configured(settings):
        raise LLMNotConfiguredError(
            f"Clé API manquante : ajoutez {API_KEY_ENV[provider]}=... dans le fichier .env "
            f"(fournisseur actuel : {provider})."
        )
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model or settings.llm_model, model_provider=provider, temperature=settings.llm_temperature
    )


def estimate_cost(settings: Settings, model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Coût en USD d'un appel, ou None si le prix du modèle est inconnu."""
    if settings.llm_price_input_per_mtok is not None and settings.llm_price_output_per_mtok is not None:
        prix = (settings.llm_price_input_per_mtok, settings.llm_price_output_per_mtok)
    elif model in PRICES_PER_MTOK:
        prix = PRICES_PER_MTOK[model]
    else:
        return None
    return (input_tokens * prix[0] + output_tokens * prix[1]) / 1_000_000
