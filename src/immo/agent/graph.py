"""Agent LangGraph : garde-fou d'entrée → LLM ⇄ outils → contrôle des chiffres.

    START → guard ─(bloqué)→ END
              │
              ▼
            agent ⇄ tools          (le LLM appelle les outils autant que nécessaire)
              │
              ▼
            verify ─(chiffre non sourcé, 1 seule relance)→ agent
              │
              ▼
             END
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from immo.agent.data import DataContext
from immo.agent.guardrails import check_input, find_unverified_numbers
from immo.agent.llm import estimate_cost
from immo.agent.rag import HybridRetriever
from immo.agent.tools import build_tools
from immo.config import Settings

CONTROL_PREFIX = "[Contrôle automatique des chiffres]"
MAX_VERIFICATION_RETRIES = 1

SYSTEM_PROMPT = """Tu es « Immo IA », l'assistant du marché immobilier d'Île-de-France.
Tu réponds à partir des données DVF (ventes de logements enregistrées par la DGFiP), nettoyées dans un \
entrepôt DuckDB : {nb_ventes} ventes du {date_min} au {date_max}{avertissement_source}.

Règles :
1. Chaque chiffre de ta réponse doit provenir d'un outil appelé pendant cet échange. N'invente jamais un \
prix, une variation ou un nombre de ventes. Si la donnée n'existe pas, dis-le.
2. Évolution des prix d'une commune : tendance_prix. Autres questions chiffrées (classements, \
départements, volumes) : requete_sql. Estimation d'un bien : estimer_prix. Définitions, méthodologie, \
limites des données : recherche_documentation.
3. Fais les calculs (moyennes, écarts, pourcentages) dans la requête SQL, pas de tête.
4. Précise l'année et le nombre de ventes derrière un prix. Signale la fragilité sous 30 ventes, et \
une très forte fragilité sous 10.
5. Pour une estimation, donne le prix, la fourchette à 80 % et rappelle qu'elle est indicative.
6. Si la question ne porte pas sur l'immobilier en Île-de-France, décline en une phrase et propose ce \
que tu sais faire.
7. Réponds en français, en 5 phrases maximum sauf demande de détail. Cite tes sources entre crochets : \
[SQL], [tendance_prix], [modèle LightGBM] ou le nom du fichier de documentation.

Paris : chaque arrondissement est une commune (ex. nom_commune = 'Paris 11e Arrondissement', \
code_commune = '75111').

Schéma de l'entrepôt :
{schema}

Exemple : prix médian des appartements d'une commune par année
select annee, nb_ventes, prix_m2_median from gold.prix_m2_commune_annee
where nom_commune = 'Clichy' and type_local = 'Appartement' order by annee
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    history_len: int
    blocked: bool
    retries: int
    unverified: list[str]


def message_text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def build_system_prompt(ctx: DataContext) -> str:
    meta = ctx.metadata
    avertissement = (
        " (ATTENTION : jeu de données SYNTHÉTIQUE de démonstration, pas de vraies ventes)"
        if meta["source_donnees"] == "sample" else ""
    )
    return SYSTEM_PROMPT.format(
        nb_ventes=meta["nb_ventes"], date_min=meta["date_min"], date_max=meta["date_max"],
        avertissement_source=avertissement, schema=ctx.schema_description(),
    )


def build_graph(llm: BaseChatModel, tools: list, system_prompt: str):
    llm_with_tools = llm.bind_tools(tools)

    def guard(state: AgentState) -> dict:
        verdict = check_input(message_text(state["messages"][-1]))
        if not verdict.ok:
            return {"blocked": True, "messages": [AIMessage(content=verdict.message)]}
        return {"blocked": False}

    def agent(state: AgentState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    def verify(state: AgentState) -> dict:
        messages = state["messages"]
        history, current = messages[: state["history_len"]], messages[state["history_len"]:]
        tool_outputs = [message_text(m) for m in current if isinstance(m, ToolMessage)]
        known = [message_text(m) for m in history] + [
            message_text(m) for m in current
            if isinstance(m, HumanMessage) and not message_text(m).startswith(CONTROL_PREFIX)
        ]
        unverified = find_unverified_numbers(message_text(messages[-1]), tool_outputs, known)
        if unverified and state["retries"] < MAX_VERIFICATION_RETRIES:
            consigne = (
                f"{CONTROL_PREFIX} Ces nombres n'apparaissent dans aucun résultat d'outil : "
                f"{', '.join(unverified)}. Vérifie-les avec un outil ou retire-les, puis redonne ta "
                "réponse complète."
            )
            return {"retries": state["retries"] + 1, "unverified": unverified,
                    "messages": [HumanMessage(content=consigne)]}
        return {"unverified": unverified}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"

    def after_verify(state: AgentState) -> str:
        last = state["messages"][-1]
        return "agent" if isinstance(last, HumanMessage) else END

    graph = StateGraph(AgentState)
    graph.add_node("guard", guard)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("verify", verify)
    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", lambda s: END if s["blocked"] else "agent", ["agent", END])
    graph.add_conditional_edges("agent", after_agent, ["tools", "verify"])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("verify", after_verify, ["agent", END])
    return graph.compile()


@dataclass
class AgentAnswer:
    reponse: str
    outils: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    chiffres_non_verifies: list[str] = field(default_factory=list)
    bloque: bool = False
    tokens_entree: int = 0
    tokens_sortie: int = 0
    cout_usd: float | None = None
    latence_s: float = 0.0
    modele: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


_SOURCES_OUTILS = {
    "requete_sql": "SQL (entrepôt DuckDB)",
    "tendance_prix": "gold.prix_m2_commune_annee",
    "estimer_prix": "modèle LightGBM",
}


class ImmoAgent:
    def __init__(self, llm: BaseChatModel, ctx: DataContext, retriever: HybridRetriever,
                 settings: Settings, model_name: str | None = None):
        self.settings = settings
        self.model_name = model_name or settings.llm_model
        self.graph = build_graph(llm, build_tools(ctx, retriever), build_system_prompt(ctx))

    def ask(self, question: str, history: list[dict] | None = None) -> AgentAnswer:
        debut = time.perf_counter()
        historique = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in (history or [])[-6:]
        ]
        state = {"messages": [*historique, HumanMessage(content=question)],
                 "history_len": len(historique), "blocked": False, "retries": 0, "unverified": []}
        try:
            final = self.graph.invoke(state, {"recursion_limit": 2 * self.settings.agent_max_steps + 6})
        except GraphRecursionError:
            return AgentAnswer(
                reponse="Je n'ai pas réussi à conclure en un nombre raisonnable d'étapes. "
                        "Pouvez-vous reformuler ou préciser votre question ?",
                latence_s=round(time.perf_counter() - debut, 2), modele=self.model_name,
            )
        return self._answer(final, debut)

    def _answer(self, final: dict, debut: float) -> AgentAnswer:
        nouveaux = final["messages"][final["history_len"]:]
        outils, sources = [], []
        tokens_in = tokens_out = 0
        for m in nouveaux:
            if isinstance(m, AIMessage):
                for call in m.tool_calls:
                    outils.append({"nom": call["name"], "arguments": call["args"]})
                    if call["name"] in _SOURCES_OUTILS:
                        sources.append(_SOURCES_OUTILS[call["name"]])
                usage = m.usage_metadata or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
            elif isinstance(m, ToolMessage) and m.name == "recherche_documentation":
                try:
                    sources += [hit["source"] for hit in json.loads(message_text(m))]
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        return AgentAnswer(
            reponse=message_text(final["messages"][-1]),
            outils=outils,
            sources=list(dict.fromkeys(sources)),
            chiffres_non_verifies=final.get("unverified", []),
            bloque=final.get("blocked", False),
            tokens_entree=tokens_in,
            tokens_sortie=tokens_out,
            cout_usd=estimate_cost(self.settings, self.model_name, tokens_in, tokens_out),
            latence_s=round(time.perf_counter() - debut, 2),
            modele=self.model_name,
        )
