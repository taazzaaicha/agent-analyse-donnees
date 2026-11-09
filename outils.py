import pandas as pd


def charger(fichier):
    """Lit un fichier CSV et renvoie le DataFrame avec un court résumé."""
    df = pd.read_csv(fichier)
    resume = {
        "lignes": df.shape[0],
        "colonnes": df.shape[1],
        "noms_colonnes": list(df.columns),
    }
    return df, resume
