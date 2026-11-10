import pandas as pd
import streamlit as st

from outils import (
    charger,
    diagnostiquer,
    nettoyer,
    normaliser,
    modeliser,
    valider,
    tableau_de_bord,
    interpreter_diagnostic,
    interpreter_validation,
)
from agent import demander_agent


def preparer(df, cible):
    """Nettoie les donnees et transforme une cible texte (oui/non...) en 0 et 1."""
    d = nettoyer(df)
    if not pd.api.types.is_numeric_dtype(d[cible]):
        d[cible] = d[cible].astype("category").cat.codes
    return d


def executer_etape(etape, df, cible):
    d = preparer(df, cible)
    if etape == "Diagnostic":
        st.subheader("Interpretation")
        for phrase in interpreter_diagnostic(diagnostiquer(df)):
            st.write("- " + phrase)
    elif etape == "Nettoyage":
        st.success("Zeros impossibles remplaces par des valeurs manquantes.")
        st.dataframe(nettoyer(df).isna().sum().to_frame("manquants"))
    elif etape == "Normalisation":
        st.dataframe(normaliser(d, cible).describe().round(2))
    elif etape == "Modelisation":
        st.json(modeliser(d, cible))
    elif etape == "Validation croisee":
        resultat = valider(d, cible)
        st.subheader("Interpretation")
        st.write(interpreter_validation(resultat))
        st.json(resultat)
    elif etape == "Tableau de bord":
        for figure in tableau_de_bord(d, cible):
            st.plotly_chart(figure)


st.set_page_config(page_title="Agent d'analyse de donnees", layout="wide")
st.title("Agent d'analyse de donnees")
st.write("Deposez un fichier CSV, choisissez la colonne a predire, puis lancez une etape ou posez une question a l'agent.")

fichier = st.file_uploader("Fichier CSV", type=["csv"])

if fichier is None:
    st.info("Deposez un fichier CSV pour commencer.")
    st.stop()

chemin = "donnees_temporaires.csv"
with open(chemin, "wb") as f:
    f.write(fichier.getvalue())

if st.session_state.get("nom") != fichier.name:
    df, resume = charger(chemin)
    st.session_state["df"] = df
    st.session_state["resume"] = resume
    st.session_state["nom"] = fichier.name

df = st.session_state["df"]
resume = st.session_state["resume"]
st.write(f"{resume['lignes']} lignes, {resume['colonnes']} colonnes.")

cible = st.selectbox(
    "Colonne a predire (cible)",
    list(df.columns),
    index=len(df.columns) - 1,
)

nb_valeurs = df[cible].nunique()
if nb_valeurs != 2:
    st.warning(
        f"La colonne '{cible}' contient {nb_valeurs} valeurs differentes. "
        "Les modeles demandent exactement 2 valeurs (par exemple oui/non). "
        "Choisissez une autre colonne pour ces etapes."
    )

st.divider()
st.subheader("Etape a lancer manuellement")
etape = st.selectbox(
    "Choisissez une etape",
    ["Diagnostic", "Nettoyage", "Normalisation", "Modelisation", "Validation croisee", "Tableau de bord"],
)

if st.button("Lancer l'etape"):
    try:
        executer_etape(etape, df, cible)
    except Exception as e:
        st.error(f"Cette etape n'a pas fonctionne : {e}")

st.divider()
st.subheader("Demander a l'agent")
demande = st.text_input(
    "Votre demande en francais",
    placeholder="ex. nettoie les donnees, montre le tableau de bord et dis-moi si le modele est fiable",
)

if st.button("Envoyer a l'agent") and demande:
    with st.spinner("L'agent reflechit..."):
        try:
            texte, figures = demander_agent(demande, preparer(df, cible), cible)
            st.write(texte)
            for figure in figures:
                st.plotly_chart(figure)
        except Exception as e:
            st.error(f"Erreur de l'agent : {e}")
