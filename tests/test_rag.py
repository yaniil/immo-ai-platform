import numpy as np

from immo.agent.rag import HybridRetriever, load_chunks, reciprocal_rank_fusion, tokenize


def test_decoupage_par_section(settings):
    chunks = load_chunks(settings.docs_dir)
    assert len(chunks) > 25
    assert all(c.texte and " > " in c.titre for c in chunks)


def test_recherche_lexicale(retriever):
    assert retriever.mode == "lexical (BM25)"
    sources = [h.chunk.source for h in retriever.search("Les loyers sont-ils dans DVF ?")]
    assert "01_dvf_presentation.md" in sources
    assert retriever.search("c'est quoi une VEFA")[0].chunk.titre == "Glossaire > VEFA"


def test_fusion_rrf():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert max(scores, key=scores.get) == 1  # bien classé dans les deux listes
    assert scores[3] > scores[2]


class EmbedderFactice:
    """Embeddings déterministes : le passage contenant « location » est proche de « loyers »."""

    def embed(self, textes):
        return np.array([[1.0, 0.0] if ("loyer" in t.lower() or "location" in t.lower()) else [0.0, 1.0]
                         for t in textes])


def test_recherche_hybride(settings):
    chunks = load_chunks(settings.docs_dir)
    hybride = HybridRetriever(chunks, EmbedderFactice())
    assert hybride.mode.startswith("hybride")
    assert any("loyer" in h.chunk.texte.lower() for h in hybride.search("prix des locations"))


def test_tokenisation():
    assert tokenize("Les Ventes à Évry") == ["vente", "evry"]
