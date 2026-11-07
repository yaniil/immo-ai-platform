"""Accès en lecture à l'entrepôt DuckDB : requêtes, communes, tendances et estimation."""

from __future__ import annotations

import difflib
import json
import re
import threading
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd

from immo.agent.sql_guard import limit_rows, validate_sql
from immo.ml.predict import PriceModel

TYPES_LOCAL = ("Appartement", "Maison")


class DataNotReadyError(FileNotFoundError):
    pass


class CommuneNotFoundError(LookupError):
    def __init__(self, recherche: str, suggestions: list[str]):
        self.recherche = recherche
        self.suggestions = suggestions
        detail = f" Suggestions : {', '.join(suggestions)}." if suggestions else ""
        super().__init__(f"Commune introuvable : « {recherche} ».{detail}")


@dataclass(frozen=True)
class Commune:
    code_commune: str
    nom_commune: str
    code_departement: str
    code_postal: str | None
    longitude: float | None
    latitude: float | None
    nb_ventes: int

    def as_dict(self) -> dict:
        return asdict(self)


def normalize(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", texte).strip()


_PARIS_ARRONDISSEMENT = re.compile(r"^paris\s*(\d{1,2})\s*(?:e|er|eme|ieme)?(?:\s*arrondissement)?$")


class DataContext:
    """Point d'accès unique aux données pour l'agent et l'API.

    Une connexion DuckDB courte est ouverte à chaque requête : ainsi l'API ne bloque jamais
    la reconstruction de l'entrepôt par dbt (DuckDB n'autorise qu'un seul processus écrivain).
    """

    def __init__(self, warehouse_path: Path, model_dir: Path | None = None):
        if not warehouse_path.exists():
            raise DataNotReadyError(
                f"Entrepôt introuvable ({warehouse_path}). Lancez : immo pipeline --source dvf"
            )
        self.warehouse_path = warehouse_path
        self.model_dir = model_dir
        self._model: PriceModel | None = None
        meta = self.query_df("select * from gold.metadata").iloc[0].to_dict()
        self.metadata = {
            "source_donnees": meta["source_donnees"],
            "date_min": pd.Timestamp(meta["date_min"]).date().isoformat(),
            "date_max": pd.Timestamp(meta["date_max"]).date().isoformat(),
            "nb_ventes": int(meta["nb_ventes"]),
            "nb_communes": int(meta["nb_communes"]),
        }
        communes = self.query_df("select * from gold.communes order by nb_ventes desc")
        self.communes = [
            Commune(
                code_commune=r.code_commune,
                nom_commune=r.nom_commune,
                code_departement=r.code_departement,
                code_postal=r.code_postal,
                longitude=None if pd.isna(r.longitude) else float(r.longitude),
                latitude=None if pd.isna(r.latitude) else float(r.latitude),
                nb_ventes=int(r.nb_ventes),
            )
            for r in communes.itertuples()
        ]
        self._par_nom = {normalize(c.nom_commune): c for c in self.communes}

    # ------------------------------------------------------------------ connexions
    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(
            str(self.warehouse_path), read_only=True, config={"enable_external_access": False}
        )

    def query_df(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Requête interne (de confiance), paramétrée."""
        with self.connect() as con:
            return con.execute(sql, params or []).df()

    def run_agent_sql(self, sql: str, max_rows: int = 100, timeout_s: float = 10.0) -> tuple[pd.DataFrame, bool]:
        """Requête écrite par le LLM : validée, plafonnée et interrompue si elle est trop longue."""
        safe_sql = limit_rows(validate_sql(sql), max_rows)
        with self.connect() as con:
            timer = threading.Timer(timeout_s, con.interrupt)
            timer.start()
            try:
                df = con.execute(safe_sql).df()
            finally:
                timer.cancel()
        return df.head(max_rows), len(df) > max_rows

    # ------------------------------------------------------------------ communes
    def resolve_commune(self, recherche: str) -> Commune:
        """Retrouve une commune par nom, code INSEE, code postal ou « Paris 11 »."""
        texte = normalize(recherche)
        texte = re.sub(r"\bste\b", "sainte", re.sub(r"\bst\b", "saint", texte))  # « St Denis »
        if not texte:
            raise CommuneNotFoundError(recherche, [])

        if m := _PARIS_ARRONDISSEMENT.match(texte):
            texte = f"751{int(m.group(1)):02d}"
        if texte.isdigit() and len(texte) == 5:
            for c in self.communes:  # triées par nombre de ventes décroissant
                if texte in (c.code_commune, c.code_postal):
                    return c
        if texte in self._par_nom:
            return self._par_nom[texte]

        commencent = [c for nom, c in self._par_nom.items() if nom.startswith(texte + " ") or nom == texte]
        if commencent:
            return commencent[0]
        proches = difflib.get_close_matches(texte, list(self._par_nom), n=1, cutoff=0.85)
        if proches:
            return self._par_nom[proches[0]]
        suggestions = difflib.get_close_matches(texte, list(self._par_nom), n=3, cutoff=0.5)
        raise CommuneNotFoundError(recherche, [self._par_nom[s].nom_commune for s in suggestions])

    def search_communes(self, recherche: str = "", limit: int = 20) -> list[Commune]:
        texte = normalize(recherche)
        if not texte:
            return self.communes[:limit]
        return [c for nom, c in self._par_nom.items() if texte in nom][:limit]

    # ------------------------------------------------------------------ indicateurs
    def trend(self, code_commune: str, type_local: str = "Appartement") -> list[dict]:
        """Prix médian au m² par année, avec les variations en pourcentage."""
        df = self.query_df(
            """
            select annee, nb_ventes, prix_m2_median, prix_m2_q1, prix_m2_q3, prix_median
            from gold.prix_m2_commune_annee
            where code_commune = ? and type_local = ?
            order by annee
            """,
            [code_commune, type_local],
        )
        if df.empty:
            return []
        df["variation_annuelle_pct"] = (df["prix_m2_median"].pct_change() * 100).round(1)
        df["variation_depuis_debut_pct"] = ((df["prix_m2_median"] / df["prix_m2_median"].iloc[0] - 1) * 100).round(1)
        return json.loads(df.to_json(orient="records", force_ascii=False))  # NaN -> None, types natifs

    @property
    def model(self) -> PriceModel:
        if self._model is None:
            if self.model_dir is None:
                raise DataNotReadyError("Aucun dossier de modèle configuré.")
            self._model = PriceModel(self.model_dir)
        return self._model

    def estimate(
        self,
        commune: Commune,
        type_local: str,
        surface: float,
        nombre_pieces: int,
        surface_terrain: float = 0.0,
        nb_dependances: int = 0,
    ) -> dict:
        if type_local not in TYPES_LOCAL:
            raise ValueError(f"type_local doit valoir {TYPES_LOCAL}")
        estimation = self.model.estimate(
            type_local=type_local,
            surface=surface,
            nombre_pieces=nombre_pieces,
            code_commune=commune.code_commune,
            code_departement=commune.code_departement,
            longitude=commune.longitude,
            latitude=commune.latitude,
            surface_terrain=surface_terrain,
            nb_dependances=nb_dependances,
        )
        reference = self.query_df(
            """
            select annee, nb_ventes, prix_m2_median
            from gold.prix_m2_commune_annee
            where code_commune = ? and type_local = ?
            order by annee desc limit 1
            """,
            [commune.code_commune, type_local],
        )
        if not reference.empty:
            ref = reference.iloc[0]
            estimation["reference_commune"] = {
                "annee": int(ref.annee),
                "prix_m2_median": float(ref.prix_m2_median),
                "nb_ventes": int(ref.nb_ventes),
            }
        estimation["commune"] = commune.nom_commune
        estimation["donnees_jusqu_au"] = str(self.metadata["date_max"])
        return estimation

    # ------------------------------------------------------------------ schéma
    def schema_description(self) -> str:
        """Description des tables silver/gold, générée à partir de la documentation dbt."""
        tables = self.query_df(
            """
            select schema_name, table_name, comment from duckdb_tables()
            where schema_name in ('silver', 'gold') and table_name not in ('ml_features', 'metadata')
            order by schema_name desc, table_name
            """
        )
        columns = self.query_df(
            """
            select schema_name, table_name, column_name, data_type, comment from duckdb_columns()
            where schema_name in ('silver', 'gold')
            order by schema_name, table_name, column_index
            """
        )
        lignes = []
        for t in tables.itertuples():
            lignes.append(f"\n{t.schema_name}.{t.table_name} : {t.comment or ''}")
            cols = columns[(columns.schema_name == t.schema_name) & (columns.table_name == t.table_name)]
            for c in cols.itertuples():
                detail = f" : {c.comment}" if c.comment else ""
                lignes.append(f"  - {c.column_name} ({c.data_type}){detail}")
        return "\n".join(lignes).strip()
