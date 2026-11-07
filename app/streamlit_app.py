"""Interface Streamlit : assistant IA, estimation d'un bien et tendances du marché.

Elle ne parle qu'à l'API (variable API_URL), comme le ferait n'importe quel client.
Lancement : streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Immo IA · Île-de-France", page_icon="🏠", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
    except requests.ConnectionError:
        st.error(f"API injoignable sur {API_URL}. Lancez : immo serve")
        st.stop()
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise RuntimeError(detail if isinstance(detail, str) else str(detail))
    return r.json()


@st.cache_data(ttl=300)
def charger_communes() -> list[dict]:
    return api("GET", "/communes", params={"limite": 2000})


def euros(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


# ------------------------------------------------------------------ en-tête
sante = api("GET", "/health")
st.title("🏠 Immo IA · marché immobilier d'Île-de-France")
if sante["statut"] != "ok":
    st.error(sante.get("detail", "Données absentes."))
    st.stop()
donnees = sante["donnees"]
st.caption(f"{donnees['nb_ventes']:,} ventes de logements du {donnees['date_min']} au {donnees['date_max']} "
           f"· source : DVF (DGFiP)".replace(",", " "))
if donnees["source_donnees"] == "sample":
    st.warning("Démonstration sur un jeu de données SYNTHÉTIQUE. Lancez `immo pipeline --source dvf` "
               "pour utiliser les vraies ventes DVF.")

with st.sidebar:
    st.subheader("État du système")
    modele = sante.get("modele_prix")
    if modele:
        st.metric("Erreur médiane du modèle (MdAPE)", f"{modele['mdape_test']:.1%}")
        st.metric("Couverture de la fourchette à 80 %", f"{modele['couverture_intervalle_80']:.0%}")
    llm = sante["llm"]
    st.write(f"LLM : `{llm['fournisseur']}` / `{llm['modele']}`")
    if not llm["configure"]:
        st.info("Ajoutez votre clé API dans `.env` pour activer l'assistant.")

onglet_chat, onglet_estimation, onglet_marche = st.tabs(["💬 Assistant", "🏷️ Estimer un bien", "📈 Marché"])

# ------------------------------------------------------------------ assistant
with onglet_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    exemples = ["Quel est le prix médian au m² à Clichy ?",
                "Quelles sont les communes les plus chères des Hauts-de-Seine ?",
                "Estime un 3 pièces de 60 m² à Versailles",
                "Pourquoi certaines ventes sont-elles exclues ?"]
    colonnes = st.columns(len(exemples))
    question_exemple = None
    for col, ex in zip(colonnes, exemples, strict=True):
        if col.button(ex, width="stretch"):
            question_exemple = ex

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Posez votre question sur l'immobilier en Île-de-France") or question_exemple
    if question:
        historique = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("L'agent interroge les données..."):
                try:
                    rep = api("POST", "/assistant", json={"question": question, "historique": historique})
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()
            st.markdown(rep["reponse"])
            if rep["chiffres_non_verifies"]:
                st.warning(f"Chiffres non vérifiés par les outils : {', '.join(rep['chiffres_non_verifies'])}")
            with st.expander("Détails : outils, sources, coût"):
                for outil in rep["outils"]:
                    st.code(f"{outil['nom']}({outil['arguments']})", language="python")
                st.write("Sources :", ", ".join(rep["sources"]) or "aucune")
                cout = f"{rep['cout_usd']:.5f} USD" if rep["cout_usd"] is not None else "inconnu"
                st.write(f"{rep['tokens_entree'] + rep['tokens_sortie']} tokens · coût : {cout} · "
                         f"{rep['latence_s']} s · modèle : {rep['modele']}")
        st.session_state.messages.append({"role": "assistant", "content": rep["reponse"]})

# ------------------------------------------------------------------ estimation
communes = charger_communes()
noms = sorted(c["nom_commune"] for c in communes)

with onglet_estimation:
    with st.form("estimation"):
        c1, c2, c3 = st.columns(3)
        commune = c1.selectbox("Commune", noms, index=noms.index("Clichy") if "Clichy" in noms else 0)
        type_local = c2.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
        surface = c3.number_input("Surface (m²)", 9, 1000, 45)
        c4, c5, c6 = st.columns(3)
        pieces = c4.number_input("Pièces principales", 1, 20, 2)
        terrain = c5.number_input("Terrain (m², maisons)", 0, 100_000, 0)
        dependances = c6.number_input("Caves / parkings", 0, 10, 0)
        envoyer = st.form_submit_button("Estimer", type="primary")
    if envoyer:
        try:
            est = api("POST", "/estimation", json={
                "commune": commune, "type_local": type_local, "surface": surface,
                "nombre_pieces": pieces, "surface_terrain": terrain, "nb_dependances": dependances})
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Prix estimé", euros(est["prix_estime"]))
            m2.metric("Fourchette à 80 %", f"{est['fourchette_basse'] / 1000:.0f} – {est['fourchette_haute'] / 1000:.0f} k€")
            ref = est.get("reference_commune")
            delta = f"{est['prix_m2_estime'] / ref['prix_m2_median'] - 1:+.1%} vs médiane {ref['annee']}" if ref else None
            m3.metric("Prix au m² estimé", euros(est["prix_m2_estime"]), delta, delta_color="off")
            if ref:
                st.caption(f"Médiane de la commune en {ref['annee']} : {euros(ref['prix_m2_median'])}/m² "
                           f"sur {ref['nb_ventes']} ventes. Estimation indicative (modèle LightGBM).")

# ------------------------------------------------------------------ marché
with onglet_marche:
    c1, c2 = st.columns([3, 1])
    commune_m = c1.selectbox("Commune ", noms, index=noms.index("Clichy") if "Clichy" in noms else 0)
    type_m = c2.radio("Type", ["Appartement", "Maison"], horizontal=True)
    serie = api("GET", f"/communes/{commune_m}/tendance", params={"type_local": type_m})
    df = pd.DataFrame(serie["par_annee"])
    if df.empty:
        st.info("Pas de ventes pour ce type de bien dans cette commune.")
    else:
        fig = px.line(df, x="annee", y="prix_m2_median", markers=True,
                      labels={"annee": "Année", "prix_m2_median": "Prix médian au m² (€)"},
                      title=f"{serie['commune']['nom_commune']} · {type_m.lower()}s")
        fig.update_traces(name="Médiane", showlegend=True)
        fig.add_scatter(x=df["annee"], y=df["prix_m2_q1"], mode="lines", line={"dash": "dot"}, name="1er quartile")
        fig.add_scatter(x=df["annee"], y=df["prix_m2_q3"], mode="lines", line={"dash": "dot"}, name="3e quartile")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df.rename(columns={"annee": "Année", "nb_ventes": "Ventes", "prix_m2_median": "Médiane €/m²",
                                        "prix_m2_q1": "Q1", "prix_m2_q3": "Q3", "prix_median": "Prix médian",
                                        "variation_annuelle_pct": "Variation annuelle %",
                                        "variation_depuis_debut_pct": "Depuis le début %"}),
                     hide_index=True, width="stretch")
