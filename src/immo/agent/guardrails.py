"""Garde-fous de l'agent.

- En entrée : rejet des questions vides, trop longues ou contenant une tentative d'injection de prompt.
- En sortie : contrôle anti-hallucination des chiffres. Chaque nombre de la réponse doit provenir
  d'un résultat d'outil (ou de la question elle-même) ; sinon l'agent doit corriger sa réponse.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

MAX_QUESTION_LENGTH = 1000

INJECTION_PATTERNS = [
    r"ignore[rsz]?\b.{0,40}\b(instructions?|consignes?|r[eè]gles?|prompt)",
    r"(oublie|forget)\b.{0,40}\b(instructions?|consignes?|r[eè]gles?)",
    r"(prompt|message|instructions?)\s+(syst[eè]me|system)",
    r"system\s*prompt",
    r"\b(tu es|you are)\s+(maintenant|d[eé]sormais|now)\b",
    r"\bjailbreak\b",
    r"\bmode\s+(d[ée]veloppeur|developer|DAN)\b",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    raison: str | None = None
    message: str | None = None


def check_input(question: str) -> GuardResult:
    texte = (question or "").strip()
    if not texte:
        return GuardResult(False, "vide", "Posez-moi une question sur le marché immobilier d'Île-de-France.")
    if len(texte) > MAX_QUESTION_LENGTH:
        return GuardResult(
            False, "trop_longue",
            f"Votre question est trop longue ({len(texte)} caractères, maximum {MAX_QUESTION_LENGTH}).",
        )
    if _INJECTION.search(texte):
        return GuardResult(
            False, "injection",
            "Je ne peux pas modifier mes consignes. Je peux en revanche répondre à vos questions sur les "
            "prix de l'immobilier en Île-de-France ou estimer un bien.",
        )
    return GuardResult(True)


# ---------------------------------------------------------------------------- chiffres

# Nombres écrits à la française : « 7 350 », « 7 350,5 », « 1,2 million », « 350 k€ »
_NOMBRE_FR = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[,.](\d+))?"
    r"(?:\s*(milliards?|millions?|Md€|M€|k€))?",
    re.IGNORECASE,
)
# Nombres dans les sorties d'outils (JSON / CSV) : « 7350.0 », « -4.2 »
_NOMBRE_MACHINE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

_MULTIPLICATEURS = {"milliard": 1e9, "md€": 1e9, "million": 1e6, "m€": 1e6, "k€": 1e3}


def _multiplicateur(suffixe: str | None) -> float:
    if not suffixe:
        return 1.0
    s = suffixe.lower().rstrip("s")
    return next((v for k, v in _MULTIPLICATEURS.items() if s.startswith(k)), 1.0)


def extract_numbers_fr(texte: str) -> list[tuple[str, float]]:
    """Renvoie (texte affiché, valeur) pour chaque nombre d'un texte en français."""
    nombres = []
    for m in _NOMBRE_FR.finditer(texte):
        entier = re.sub(r"[   ]", "", m.group(1))
        valeur = float(f"{entier}.{m.group(2)}" if m.group(2) else entier)
        nombres.append((m.group(0).strip(), valeur * _multiplicateur(m.group(3))))
    return nombres


def extract_numbers_machine(texte: str) -> list[float]:
    return [float(x) for x in _NOMBRE_MACHINE.findall(texte)]


def _a_verifier(valeur: float) -> bool:
    if valeur.is_integer() and 1900 <= valeur <= 2100:  # années
        return False
    return abs(valeur) > 12  # petits entiers : pièces, mois, rangs...


def find_unverified_numbers(
    answer: str,
    tool_outputs: Iterable[str],
    known_texts: Iterable[str] = (),
    tolerance: float = 0.015,
) -> list[str]:
    """Liste les nombres de la réponse qui n'apparaissent dans aucune source (tolérance d'arrondi 1,5 %)."""
    sources: list[float] = []
    for texte in tool_outputs:
        sources += [abs(v) for v in extract_numbers_machine(texte)]
        sources += [abs(v) for _, v in extract_numbers_fr(texte)]
    for texte in known_texts:
        sources += [abs(v) for _, v in extract_numbers_fr(texte)]

    non_verifies = []
    for affiche, valeur in extract_numbers_fr(answer):
        v = abs(valeur)
        if not _a_verifier(v):
            continue
        if not any(abs(v - s) <= max(tolerance * s, 0.51) for s in sources):
            non_verifies.append(affiche)
    return list(dict.fromkeys(non_verifies))  # sans doublons, ordre conservé
