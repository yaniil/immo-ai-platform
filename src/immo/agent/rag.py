"""RAG hybride sur la documentation du projet (docs/knowledge).

Deux moteurs de recherche complémentaires, fusionnés par Reciprocal Rank Fusion (RRF) :
- BM25 (lexical) : excellent sur les termes exacts (« VEFA », « Carrez », « MdAPE ») ;
- embeddings (sémantique, via fastembed, en local) : retrouve une reformulation
  (« loyers » ↔ « locations »). Optionnel : sans fastembed, la recherche reste lexicale.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

STOPWORDS = set(
    "a au aux avec ce ces cet cette d dans de des du elle en est et etre il ils je l la le les leur lui "
    "ma mais me meme mes mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sont sur ta "
    "te tes toi ton tu un une vos votre vous y c est quel quelle quels quelles comment pourquoi".split()
)


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    titre: str
    texte: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class Embedder(Protocol):
    def embed(self, textes: list[str]) -> np.ndarray: ...


def tokenize(texte: str) -> list[str]:
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode().lower()
    tokens = re.findall(r"[a-z0-9]+", texte)
    # racinisation minimale : « ventes » et « vente » deviennent le même mot
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens if t not in STOPWORDS]


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """Découpe chaque fichier Markdown par section de niveau 2 (## Titre)."""
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        document = ""
        for bloc in re.split(r"^## ", path.read_text("utf-8"), flags=re.MULTILINE):
            if bloc.startswith("# "):  # titre du document, avant la première section
                document = bloc.splitlines()[0][2:].strip()
                continue
            titre, _, corps = bloc.partition("\n")
            if corps.strip():
                chunks.append(Chunk(
                    id=f"{path.stem}#{len(chunks)}",
                    source=path.name,
                    titre=f"{document} > {titre.strip()}",
                    texte=corps.strip(),
                ))
    return chunks


def reciprocal_rank_fusion(classements: list[list[int]], k: int = 60) -> dict[int, float]:
    """Fusionne plusieurs classements : score = somme des 1 / (k + rang)."""
    scores: dict[int, float] = {}
    for classement in classements:
        for rang, idx in enumerate(classement):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rang + 1)
    return scores


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed(self, textes: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(textes)), dtype=np.float32)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        if not chunks:
            raise ValueError("Aucun document à indexer.")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(f"{c.titre} {c.texte}") for c in chunks])
        self.embedder = embedder
        self.vectors = None
        if embedder is not None:
            self.vectors = self._normalize(embedder.embed([f"{c.titre}\n{c.texte}" for c in chunks]))

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)

    @property
    def mode(self) -> str:
        return "hybride (BM25 + embeddings)" if self.vectors is not None else "lexical (BM25)"

    def search(self, question: str, k: int = 4, candidats: int = 10) -> list[SearchHit]:
        bm25_scores = self.bm25.get_scores(tokenize(question))
        classements = [[int(i) for i in np.argsort(-bm25_scores)[:candidats] if bm25_scores[i] > 0]]
        if self.vectors is not None:
            q = self._normalize(self.embedder.embed([question]))[0]
            classements.append([int(i) for i in np.argsort(-(self.vectors @ q))[:candidats]])
        fusion = reciprocal_rank_fusion(classements)
        meilleurs = sorted(fusion.items(), key=lambda x: -x[1])[:k]
        return [SearchHit(self.chunks[i], round(score, 4)) for i, score in meilleurs]


def build_retriever(docs_dir: Path, dense: bool = True, embedding_model: str | None = None) -> HybridRetriever:
    chunks = load_chunks(docs_dir)
    embedder = None
    if dense and embedding_model:
        try:
            embedder = FastEmbedEmbedder(embedding_model)
        except Exception as exc:  # fastembed absent ou modèle non téléchargeable
            logger.warning("Recherche sémantique désactivée (%s) : BM25 seul.", exc)
    retriever = HybridRetriever(chunks, embedder)
    logger.info("Index RAG : %d passages, mode %s", len(chunks), retriever.mode)
    return retriever
