"""Outils mis à disposition de l'agent. Chaque outil renvoie du JSON (ou du CSV) que le LLM lit."""

from __future__ import annotations

import json
from typing import Literal

import duckdb
from langchain_core.tools import BaseTool, tool

from immo.agent.data import CommuneNotFoundError, DataContext
from immo.agent.rag import HybridRetriever
from immo.agent.sql_guard import UnsafeSQLError

TypeLocal = Literal["Appartement", "Maison"]


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def build_tools(ctx: DataContext, retriever: HybridRetriever) -> list[BaseTool]:
    @tool
    def requete_sql(sql: str) -> str:
        """Exécute une requête SQL DuckDB en LECTURE SEULE sur l'entrepôt et renvoie le résultat en CSV.

        Utilise-la pour les questions chiffrées : prix médians, nombres de ventes, classements,
        comparaisons. Tables disponibles : voir le schéma dans tes instructions (préfixe gold. ou silver.).
        Une seule requête SELECT, 100 lignes maximum. Préfère les tables gold (déjà agrégées).
        """
        try:
            df, tronque = ctx.run_agent_sql(sql)
        except UnsafeSQLError as exc:
            return f"REQUÊTE REFUSÉE : {exc}"
        except duckdb.Error as exc:
            return f"ERREUR SQL : {exc}. Corrige la requête (vérifie les noms de tables et de colonnes)."
        if df.empty:
            return "Aucun résultat."
        csv = df.to_csv(index=False)
        return csv + ("\n(résultat tronqué à 100 lignes)" if tronque else "")

    @tool
    def tendance_prix(commune: str, type_local: TypeLocal = "Appartement") -> str:
        """Évolution annuelle du prix médian au m² dans une commune, avec les variations en %.

        `commune` accepte un nom (« Clichy », « Paris 11 »), un code INSEE ou un code postal.
        """
        try:
            c = ctx.resolve_commune(commune)
        except CommuneNotFoundError as exc:
            return f"ERREUR : {exc}"
        return _json({"commune": c.nom_commune, "code_commune": c.code_commune, "type_local": type_local,
                      "par_annee": ctx.trend(c.code_commune, type_local)})

    @tool
    def estimer_prix(
        commune: str,
        surface: float,
        nombre_pieces: int,
        type_local: TypeLocal = "Appartement",
        surface_terrain: float = 0.0,
        nb_dependances: int = 0,
    ) -> str:
        """Estime le prix de vente d'un logement avec le modèle LightGBM (prix, prix au m², fourchette à 80 %).

        `surface` en m² (surface habitable), `surface_terrain` en m² pour une maison,
        `nb_dependances` = nombre de caves ou parkings inclus.
        """
        if not 9 <= surface <= 1000:
            return "ERREUR : la surface doit être comprise entre 9 et 1000 m²."
        try:
            c = ctx.resolve_commune(commune)
        except CommuneNotFoundError as exc:
            return f"ERREUR : {exc}"
        return _json(ctx.estimate(c, type_local, surface, max(int(nombre_pieces), 1),
                                  surface_terrain, nb_dependances))

    @tool
    def recherche_documentation(question: str) -> str:
        """Cherche dans la documentation du projet : définitions (VEFA, Carrez...), méthodologie de nettoyage,
        contenu et limites des données DVF, fonctionnement et limites du modèle d'estimation.
        Cite la source renvoyée dans ta réponse.
        """
        hits = retriever.search(question, k=4)
        return _json([{"source": h.chunk.source, "section": h.chunk.titre, "texte": h.chunk.texte} for h in hits])

    return [requete_sql, tendance_prix, estimer_prix, recherche_documentation]
