import pandas as pd

"""
--------------------
CARICAMENTO DATASET
--------------------
"""

def caricamento_dataset (path):
    try:
        df = pd.read_csv(path)
        print("\nDataset caricato con successo.\n")
        return df
    except FileNotFoundError:
        print("Errore: Il file del dataset non è stato trovato")
        return None
