# MLP vs CNN for Intrusion Detection System (IDS)

Questo repository presenta uno studio comparativo tra un'architettura **Multi-Layer Perceptron (MLP)** e una **Convolutional Neural Network (CNN) 1D**. 

Il progetto affronta un problema di **Supervised Learning** (Apprendimento Supervisionato) mirato alla **Classificazione Binaria** (Traffico Normale vs. Attacco), valutando la capacità dei modelli di identificare tentativi di intrusione all'interno di un traffico di rete rappresentato da dati tabulari.

## 📁 Struttura del Progetto

Il software è stato sviluppato in ambiente locale tramite l'IDE **PyCharm**, adottando una struttura modulare per separare la logica dei dati dall'architettura dei modelli:

- `main.py`: Lo script principale che coordina l'intera pipeline: caricamento dati, analisi esplorativa, preprocessing, addestramento e valutazione finale.
- `moduli/`: Core logico del sistema:
  - `data_loader.py`: Funzioni per il caricamento sicuro del dataset CSV.
  - `model.py`: Definisce le classi PyTorch `RilevatoreAttacchi` (MLP) e `RilevatoreAttacchiCNN`.
  - `preprocessing.py`: Gestisce l'identificazione automatica delle feature e la rimozione di ID non informativi (es. `session_id`).
  - `visualizzazione.py`: Strumenti grafici per monitorare la stabilità dell'addestramento tramite media e deviazione standard su 5 Fold.
- `dataset/`: Cartella destinata al file `cybersecurity_intrusion_data.csv`.

## ⚙️ Caratteristiche Tecniche

### Preprocessing & Engineering
- **Global Seed (42)**: È stato fissato un seme di casualità unico. Questo garantisce che i risultati siano **100% riproducibili**: i pesi iniziali dei neuroni e la divisione dei dati saranno identici a ogni esecuzione.
- **Gestione Valori Mancanti**: I valori nulli in `encryption_used` vengono trattati come categoria "Unknown".
- **Trasformazione**: Utilizzo di `ColumnTransformer` per applicare `StandardScaler` (features numeriche) e `OneHotEncoder` (features categoriche).
- **Bilanciamento**: Utilizzo di split stratificati per mantenere la proporzione delle classi con l'uso di `stratify`.

### Architetture dei Modelli
- **MLP**: Rete densa con strati da 64 e 32 neuroni e attivazione ReLU.
- **CNN 1D**: Modello che interpreta le feature tabulari come una sequenza spaziale, utilizzando convoluzioni, Batch Normalization e un livello di Adaptive Max Pooling.

### Validazione
- **5-Fold Cross-Validation**: Validazione su 5 diverse partizioni dei dati per garantire stabilità statistica.
- **Metriche**: Valutazione tramite Accuracy, Precision, Recall e F1-Score.

## ⚙️ Configurazione (Scelta del Modello)

Nel file `main.py`, la scelta del modello da addestrare avviene manualmente decommentando la riga desiderata all'interno del ciclo di Cross-Validation.

**Istruzioni:**
Aprire `main.py`, scorrere fino alla sezione dell'inizializzazione del modello e agire sulle seguenti righe:

```python
# Per testare la CNN:
model = RilevatoreAttacchiCNN(X_processed.shape[1]).to(device)
# model = RilevatoreAttacchi(X_processed.shape[1], 64, 32).to(device)

# Per testare l'MLP:
# model = RilevatoreAttacchiCNN(X_processed.shape[1]).to(device)
model = RilevatoreAttacchi(X_processed.shape[1], 64, 32).to(device)
```

## 📊 Il Dataset

Per favorire la riproducibilità, il dataset (709 KB) è incluso direttamente nella repository. Include feature tecniche etichettate per distinguere tra traffico normale e intrusioni informatiche.

## 🛠️ Installazione

Il codice richiede **Python 3.8+**. 

1. **Clonare il repository**:
   ```bash
   git clone https://github.com/ClosingChain/MLP-vs-CNN-Intrusion-Detection.git
   cd MLP-vs-CNN-Intrusion-Detection
   ```

2. Installare le librerie necessarie:
   ```bash
    pip install -r requirements.txt
   ```

## 🚀 Esecuzione

Per avviare la pipeline completa (preprocessing, addestramento dei modelli e confronto tramite 5-fold Cross-Validation), eseguire il comando:
   ```bash
    python main.py
```
