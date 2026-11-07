import pytest
from conftest import ai, scripted
from fastapi.testclient import TestClient

from immo.agent.graph import ImmoAgent
from immo.api.main import create_app


@pytest.fixture(scope="module")
def client(settings, ctx, retriever):
    def fausse_fabrique(context):
        return ImmoAgent(scripted(ai("Bonjour, je suis prêt."), ai("Toujours prêt.")), context, retriever, settings)

    return TestClient(create_app(settings, agent_factory=fausse_fabrique))


def test_health(client):
    r = client.get("/health").json()
    assert r["statut"] == "ok"
    assert r["donnees"]["source_donnees"] == "sample"
    assert 0 < r["modele_prix"]["mdape_test"] < 1


def test_recherche_de_communes(client):
    noms = [c["nom_commune"] for c in client.get("/communes", params={"recherche": "saint"}).json()]
    assert set(noms) == {"Saint-Denis", "Saint-Germain-en-Laye"}


def test_tendance(client):
    r = client.get("/communes/Paris 11/tendance").json()
    assert r["commune"]["code_commune"] == "75111"
    assert [a["annee"] for a in r["par_annee"]] == [2024, 2025]
    assert client.get("/communes/Lyon/tendance").status_code == 404


def test_estimation(client):
    r = client.post("/estimation", json={"commune": "Clichy", "surface": 45, "nombre_pieces": 2})
    assert r.status_code == 200
    e = r.json()
    assert e["fourchette_basse"] <= e["prix_estime"] <= e["fourchette_haute"]
    assert e["reference_commune"]["annee"] == 2025


@pytest.mark.parametrize("payload", [
    {"commune": "Clichy", "surface": 3, "nombre_pieces": 1},
    {"commune": "Clichy", "surface": 45, "nombre_pieces": 0},
    {"commune": "Clichy", "surface": 45, "nombre_pieces": 2, "type_local": "Château"},
])
def test_estimation_validation(client, payload):
    assert client.post("/estimation", json=payload).status_code == 422


def test_assistant(client):
    r = client.post("/assistant", json={"question": "Bonjour"}).json()
    assert r["reponse"] == "Bonjour, je suis prêt."
    assert r["tokens_entree"] == 1000


def test_assistant_sans_cle_api(settings, ctx):
    app = create_app(settings)  # vraie fabrique : aucune clé GROQ_API_KEY en test
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("GROQ_API_KEY", raising=False)
        r = TestClient(app).post("/assistant", json={"question": "Bonjour"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]
