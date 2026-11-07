"""Garde-fou SQL : l'agent ne peut exécuter que des requêtes de lecture.

Défense en profondeur :
1. validation du texte (une seule requête SELECT/WITH, aucun mot-clé d'écriture ni lecture de fichier) ;
2. connexion DuckDB en lecture seule ET sans accès externe (fichiers, réseau, extensions) ;
3. nombre de lignes plafonné et délai maximal d'exécution.
"""

from __future__ import annotations

import re

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|install|load|pragma|"
    r"set|reset|call|checkpoint|vacuum|grant|revoke|truncate|merge|use|begin|commit|rollback)\b",
    re.IGNORECASE,
)
FORBIDDEN_FUNCTIONS = re.compile(
    r"\b(read_\w+|glob|sniff_csv|parquet_\w+|query|query_table|getenv|sql_auto_complete)\s*\(",
    re.IGNORECASE,
)
FILE_SCAN = re.compile(r"\b(from|join)\s+['\"]", re.IGNORECASE)  # ex. SELECT * FROM 'fichier.csv'


class UnsafeSQLError(ValueError):
    """Requête refusée par le garde-fou."""


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", sql)


def validate_sql(sql: str) -> str:
    """Renvoie la requête nettoyée si elle est sûre, lève UnsafeSQLError sinon."""
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("Requête vide.")
    if ";" in cleaned:
        raise UnsafeSQLError("Une seule requête à la fois.")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise UnsafeSQLError("Seules les requêtes SELECT (ou WITH ... SELECT) sont autorisées.")
    # On retire le contenu des chaînes de caractères avant de chercher les mots interdits,
    # pour accepter par exemple WHERE nom_commune = 'Saint-Germain-en-Laye'.
    without_strings = re.sub(r"'(?:[^']|'')*'", "''", cleaned)
    if match := FORBIDDEN_KEYWORDS.search(without_strings):
        raise UnsafeSQLError(f"Mot-clé interdit : {match.group(1).upper()}.")
    if match := FORBIDDEN_FUNCTIONS.search(without_strings):
        raise UnsafeSQLError(f"Fonction interdite : {match.group(1)}.")
    if FILE_SCAN.search(cleaned):
        raise UnsafeSQLError("La lecture directe de fichiers est interdite.")
    return cleaned


def limit_rows(sql: str, max_rows: int) -> str:
    """Encapsule la requête pour plafonner le nombre de lignes renvoyées (+1 pour détecter la troncature)."""
    return f"select * from (\n{sql}\n) as resultat limit {max_rows + 1}"
