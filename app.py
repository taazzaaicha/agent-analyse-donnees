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

st.set_page_config(page_title="Agent d'analyse de donnees", layout="wide")
st.title("Agent d'analyse de donnees")
st.write("Deposez un fichier CSV, choisissez une etape, puis lancez-la.")

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

st.subheader("Resume du fichier")
st.write(
    f"{st.session_state['resume']['lignes']} lignes, "
    f"{st.session_state['resume']['colonnes']} colonnes."
)

st.subheader("Etape a lancer")
etape = st.selectbox(
    "Choisissez une etape",
    ["Diagnostic", "Nettoyage", "Normalisation", "Modelisation", "Validation croisee", "Tableau de bord"],
)

if st.button("Lancer l'etape"):
    if etape == "Diagnostic":
        st.subheader("Interpretation")
        for phrase in interpreter_diagnostic(diagnostiquer(df)):
            st.write("- " + phrase)

    elif etape == "Nettoyage":
        propre = nettoyer(df)
        st.success("Zeros impossibles remplaces par des valeurs manquantes.")
        st.dataframe(propre.isna().sum().to_frame("manquants"))

    elif etape == "Normalisation":
        norm = normaliser(nettoyer(df))
        st.write("Statistiques apres normalisation :")
        st.dataframe(norm.describe().round(2))

    elif etape == "Modelisation":
        st.write("Comparaison de deux modeles sur un jeu de test separe :")
        st.json(modeliser(nettoyer(df)))

    elif etape == "Validation croisee":
        resultat = valider(nettoyer(df))
        st.subheader("Interpretation")
        st.write(interpreter_validation(resultat))
        st.json(resultat)

    elif etape == "Tableau de bord":
        for figure in tableau_de_bord(nettoyer(df)):
            st.plotly_chart(figure)
        
        
