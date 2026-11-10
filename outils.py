import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import plotly.express as px



COLONNES_PIMA = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

# Colonnes où un zéro est impossible : il signifie une valeur manquante
COLONNES_ZERO_IMPOSSIBLE = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def charger(fichier):
    """Lit un fichier CSV et renvoie le DataFrame avec un court resume."""
    df = pd.read_csv(fichier)

    # Si les noms de colonnes sont des nombres, le fichier n'a pas d'en-tete
    if all(str(c).replace(".", "", 1).isdigit() for c in df.columns):
        df = pd.read_csv(fichier, header=None, names=COLONNES_PIMA)

    resume = {
        "lignes": df.shape[0],
        "colonnes": df.shape[1],
        "noms_colonnes": list(df.columns),
    }
    return df, resume


def diagnostiquer(df):
    """Renvoie un diagnostic : valeurs manquantes, doublons, zeros suspects et types."""
    zeros_suspects = {
        c: int((df[c] == 0).sum())
        for c in COLONNES_ZERO_IMPOSSIBLE
        if c in df.columns
    }
    return {
        "valeurs_manquantes": {c: int(n) for c, n in df.isna().sum().items()},
        "doublons": int(df.duplicated().sum()),
        "zeros_suspects": zeros_suspects,
        "types": {c: str(t) for c, t in df.dtypes.items()},
    }


def nettoyer(df):
    """Remplace les zeros impossibles par des valeurs manquantes (NaN)."""
    df = df.copy()
    for c in COLONNES_ZERO_IMPOSSIBLE:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    return df

def normaliser(df, cible="Outcome"):
    """Met les variables explicatives a la meme echelle (moyenne 0, ecart-type 1)."""
    df = df.copy()
    colonnes = [c for c in df.columns if c != cible]
    df[colonnes] = StandardScaler().fit_transform(df[colonnes])
    return df


def modeliser(df, cible="Outcome"):
    """Compare deux modeles sur un jeu de test separe. Renvoie l'AUC de chacun."""
    X = df.drop(columns=[cible])
    y = df[cible]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    modeles = {
        "Regression logistique": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    resultats = {}
    for nom, modele in modeles.items():
        pipeline = Pipeline([
            ("imputation", SimpleImputer(strategy="median")),
            ("normalisation", StandardScaler()),
            ("modele", modele),
        ])
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        resultats[nom] = {
            "auc_test": round(roc_auc_score(y_test, proba), 3),
            "accuracy_test": round(pipeline.score(X_test, y_test), 3),
        }
    return resultats


def valider(df, cible="Outcome", modele=None):
    """Validation croisee stratifiee a 5 plis. Renvoie la moyenne et l'ecart-type de l'AUC."""
    X = df.drop(columns=[cible])
    y = df[cible]
    if modele is None:
        modele = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("normalisation", StandardScaler()),
        ("modele", modele),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    return {
        "auc_moyenne": float(round(scores.mean(), 3)),
        "auc_ecart_type": float(round(scores.std(), 3)),
        "auc_par_pli": [float(round(s, 3)) for s in scores],
    }


def tableau_de_bord(df, cible="Outcome"):
    """Renvoie une liste de graphiques : distribution de chaque variable et correlations."""
    graphiques = []

    for c in df.columns:
        fig = px.histogram(df, x=c, color=cible, title=f"Distribution de {c}", barmode="overlay")
        graphiques.append(fig)

    corr = df.corr(numeric_only=True).round(2)
    fig_corr = px.imshow(corr, text_auto=True, title="Matrice de correlation")
    graphiques.append(fig_corr)

    return graphiques


    
def interpreter_diagnostic(diag):
    """Transforme le diagnostic en phrases en francais."""
    lignes = []
    for colonne, n in diag["zeros_suspects"].items():
        if n > 0:
            lignes.append(
                f"{colonne} contient {n} valeurs a zero, impossibles pour cette mesure. "
                "Ce sont probablement des valeurs manquantes."
            )
    if diag["doublons"] > 0:
        lignes.append(f"{diag['doublons']} lignes en double ont ete trouvees.")
    if not lignes:
        lignes.append("Aucun probleme majeur detecte.")
    return lignes


def interpreter_validation(resultat):
    """Transforme le resultat de valider en phrase en francais."""
    auc = resultat["auc_moyenne"]
    ecart = resultat["auc_ecart_type"]

    if auc >= 0.9:
        niveau = "excellente"
    elif auc >= 0.8:
        niveau = "bonne"
    elif auc >= 0.7:
        niveau = "correcte"
    else:
        niveau = "faible"

    stabilite = "stable" if ecart < 0.05 else "instable selon le decoupage des donnees"
    return (
        f"L'AUC moyenne est de {auc}, ce qui indique une discrimination {niveau} "
        f"entre les personnes malades et les autres. Le resultat est {stabilite} "
        f"(ecart-type de {ecart})."
    )
