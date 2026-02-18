import matplotlib.pyplot as plt
import numpy as np


def plotta_risultati_cv(history_list, nome_metrica="Accuracy", colore="blue"):
    """
    Crea un grafico professionale con Media e Deviazione Standard per la Cross-Validation.
    """
    # Trasformiamo la lista di liste in un array NumPy per i calcoli
    data = np.array(history_list)

    # Calcoliamo media e deviazione standard per ogni epoca (asse delle colonne)
    media = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Creiamo l'asse delle epoche
    epoche = range(1, len(media) + 1)

    plt.figure(figsize=(10, 6))

    # 1. Disegniamo la linea della media
    plt.plot(epoche, media, label=f'Media {nome_metrica}', color=colore, linewidth=2)

    # 2. Disegniamo l'area di deviazione standard (l'ombra)
    # alpha=0.2 rende il colore trasparente per l'effetto "ombra"
    plt.fill_between(epoche, media - std, media + std, color=colore, alpha=0.2, label='Stabilità (Std Dev)')

    # Personalizzazione del grafico
    plt.title(f'Analisi Stabilità {nome_metrica} su 5 Fold')
    plt.xlabel('Epoca')
    plt.ylabel(nome_metrica)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Mostra il grafico a video
    #plt.show()


def plotta_confronto_modelli(nomi_modelli, accuratezze, precisioni, recall):
    """
    Crea un grafico a barre per confrontare le prestazioni di diversi modelli.
    """
    x = np.arange(len(nomi_modelli))  # Posizioni dei modelli sull'asse X
    width = 0.25  # Larghezza delle barre

    plt.figure(figsize=(10, 6))

    # Creiamo le barre per ogni metrica
    plt.bar(x - width, accuratezze, width, label='Accuracy', color='skyblue')
    plt.bar(x, precisioni, width, label='Precision', color='lightgreen')
    plt.bar(x + width, recall, width, label='Recall', color='salmon')

    # Personalizzazione del grafico
    plt.ylabel('Punteggio (0-1)')
    plt.title('Confronto Prestazioni: Baseline MLP vs CNN')
    plt.xticks(x, nomi_modelli)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0.5, 1.0)  # Zoom sulla parte alta per vedere meglio le differenze