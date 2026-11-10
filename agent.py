import os

import streamlit as st
from google import genai
from google.genai import types

from outils import (
    diagnostiquer,
    interpreter_diagnostic,
    nettoyer,
    modeliser,
    valider,
    interpreter_validation,
    tableau_de_bord,
)

MODELE = "gemini-3.5-flash"

def obtenir_cle():
    """Lit la clé Gemini : variable d'environnement en local, Secrets en ligne."""
    cle = os.environ.get("GEMINI_API_KEY")
    if not cle:
        cle = st.secrets["GEMINI_API_KEY"]
    return cle


def construire_outils(df, figures, cible):
    """Cree les outils que Gemini peut appeler. Chaque outil renvoie du texte."""

    def diagnostic() -> str:
        """Diagnostique les donnees : valeurs a zero impossibles et doublons. A utiliser pour decrire la qualite des donnees."""
        return " ".join(interpreter_diagnostic(diagnostiquer(df)))

    def nettoyage() -> str:
        """Nettoie les donnees en remplacant les zeros impossibles par des valeurs manquantes, et indique le nombre de manquants par colonne."""
        manquants = nettoyer(df).isna().sum()
        return "Valeurs manquantes apres nettoyage : " + ", ".join(
            f"{c} : {n}" for c, n in manquants.items() if n > 0
        )

    def modelisation() -> str:
        """Compare deux modeles de prediction (regression logistique et random forest) sur un jeu de test pour predire la colonne cible."""
        resultats = modeliser(nettoyer(df), cible)
        return "; ".join(
            f"{nom} : AUC test {v['auc_test']}, precision test {v['accuracy_test']}"
            for nom, v in resultats.items()
        )

    def validation() -> str:
        """Valide le modele par validation croisee a 5 plis et donne une interpretation de sa qualite et de sa stabilite."""
        return interpreter_validation(valider(nettoyer(df), cible))

    def tableau() -> str:
        """Cree les graphiques du tableau de bord : distribution de chaque variable et matrice de correlation."""
        figures.extend(tableau_de_bord(nettoyer(df), cible))
        return f"{len(figures)} graphiques crees et affiches."

    return [diagnostic, nettoyage, modelisation, validation, tableau]


def demander_agent(demande, df, cible="Outcome"):
    """Envoie la demande a Gemini, qui appelle les outils et repond en francais."""
    figures = []
    client = genai.Client(api_key=obtenir_cle())
    config = types.GenerateContentConfig(
        system_instruction=(
            "Tu es un assistant d'analyse de donnees. Reponds toujours en francais. "
            f"La colonne a predire s'appelle '{cible}'. "
            "Utilise les outils pour obtenir les resultats avant de repondre. "
            "N'invente jamais de chiffre : appuie-toi uniquement sur ce que renvoient les outils."
        ),
        tools=construire_outils(df, figures, cible),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=5
        ),
    )
    reponse = client.models.generate_content(
        model=MODELE, contents=demande, config=config
    )
    return reponse.text or "Je n'ai pas pu produire de reponse.", figures
