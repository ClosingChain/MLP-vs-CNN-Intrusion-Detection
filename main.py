import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

from moduli.data_loader import caricamento_dataset
from moduli.model import RilevatoreAttacchi, RilevatoreAttacchiCNN
from moduli.preprocessing import split_features
from moduli.visualizzazione import plotta_risultati_cv, plotta_confronto_modelli

#Fissa la casualità per risultati riproducibili
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  #Fondamentale poichè i pesi INIZIALI dei neuroni sono sempre gli stessi
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Forza PyTorch a usare algoritmi deterministici
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
--------------------
CARICAMENTO DATASET
--------------------
"""
set_seed(42)
path_file = "dataset/cybersecurity_intrusion_data.csv"
df = caricamento_dataset(path_file)

"""
-----------------------------
ANALISI ESPLORATIVA DEI DATI
-----------------------------
"""
#Vediamo le prime 5 righe del dataset
#print(df.head(),"\n")

#Capiamo i tipi di dati (numerici, testuali) e se ci sono valori mancanti
#print(df.info(),"\n")
nan_report = df.isnull().sum()
#print(f"\n I valori mancanti sono: {nan_report[nan_report > 0]}")

# Controlliamo i duplicati escludendo la colonna session_id poichè questa colonna non trasmette nessuna informazione
# Inoltre è normale che non avremo mai duplicati se la considerassimo
duplicati = df.drop(columns=['session_id']).duplicated().sum()
#print(f"Righe con contenuto identico: {duplicati}")

#Questa è la colonna che ci interessa per il nostro problema, vediamo se contiene valori bilanciati o no.
#print("\n--- Distribuzione della variabile target ('attack_detected') ---")
print(df['attack_detected'].value_counts())

"""
-------------
DATA CLEANING
-------------
"""
# Sostituiamo i NaN con la stringa 'Unknown' essendo che le reti neurali non possono elaborare valori nulli.
df['encryption_used'] = df['encryption_used'].fillna('Unknown')

'''
#Ho pensato che avere dei NaN in questa feature "encryption_used" fosse sospetto quindi con queste 3 righe di codice vedo le percentuali
#di attacchi e non attacchi che ci sono. Sfortunatamente le percentuali rispecchiano la distribuizione del dataset con una varianza del 2.3% quindi niente di fatto
cross_tab = pd.crosstab(df['encryption_used'], df['attack_detected'], normalize='index') * 100
print("--- Analisi Correlazione Encryption ---")
print(cross_tab)
'''

"""
-------------
PREPROCESSING
-------------
"""
#Dall'eplorazione del dataset mi sono accorto che molte features sono --> Dtype = Object.
#Questo non va bene perchè il modello elaborerà solo formati di tipo numerico (int, float, etc)
#Quindi utilizzerò "StandardScaler" per le features numeriche e "OneHotEncoder" per quelle testuali(objet) per renderle numeriche.

numeric_features, object_features = split_features(df)  #    <---- Guardare modulo preprocessing.py

#Separazione delle features (X) dalla variabile target (y)
X = df.drop(['attack_detected', 'session_id'], axis=1)
y = df['attack_detected']

print("\n--- Dimensioni dei set X e y ---")
print(f"Dimensioni delle feature (X): {X.shape}")
print(f"Dimensioni della variabile target (y): {y.shape}")

#Eseguo il ColumnTransformer che prende tutte le feautures corrispondenti a "numeric_features" e applica lo StandardScaler
#Lo stesso lo farà con "object_features". handle_unknown='ignore' --> mette a '0' quando, se ci sono nuovi dati, la colonna sconosciuta
# remainder='passthrough' --> mi dice cosa fare con le colonne che non sono state specificate in nessuna delle tuple del preprocessor,
#n questo caso li mantiene inalterate.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('obj', OneHotEncoder(handle_unknown='ignore'), object_features)
    ],
    remainder='passthrough'
)

# Adesso applichiamo il preprocessor al nostro set di feature X.
X_processed = preprocessor.fit_transform(X)
print(f"Dimensioni di X_processed: {X_processed.shape}")
print(f"Tipo di dato di X_processed: {X_processed.dtype}")
print(f"Prime 5 righe di X_processed:\n: {X_processed[:5]}")

"""
-----------------------------
PREPARAZIONE DATI PER PYTORCH 
-----------------------------
"""
#Per prima cosa splitto il mio dataset
#Isoliamo il TEST SET (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_processed, y,
    test_size=0.2, # Il 20% dei dati verrà utilizzato come set di test
    random_state=42, # Per garantire la riproducibilità dei risultati
    stratify=y #Se abbiamo il 45% di attacchi nel dataset completo, avremo circa il 45% di attacchi sia nel y_train che nel y_test.
)

'''# Secondo split: dividiamo l'80% in 60% training e 20% validation
# Usiamo test_size=0.25 perché 0.25 * 0.80 = 0.20 del totale
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)'''

# PREPARAZIONE DEL TEST LOADER
#Trasformazione degli array Numpy in tensori. il .float() è fondamentale poichè i pesi delle reti neurali si aspettano dati in float32.
#torch.from_numpy == torch.tensor() con la differenza che con la seconda creo una copia dell'array numpy corrispondente in tensore con la prima no.
#Quindi risparmio in memoria.
X_test_tensore = torch.from_numpy(X_test).float()
y_test_tensore = torch.from_numpy(y_test.values).float()

#Creiamo il TensorDataset(Dataset) fondamentale per tenere incollata ogni caratteristica X alla sua etichetta Y corrispondente
test_dataset = TensorDataset(X_test_tensore, y_test_tensore)

# (DataLoader) che preleva i dati dal dataset e li divide in batch
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

#Prepariamo la K-Fold che garantisce la robustezza del mio modello.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
risultati_val_accuracy = []

# Controlliamo se la GPU è disponibile, altrimenti usiamo la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Salvataggio per visualizzazione grafici
history_train_loss = []
history_val_loss = []
history_val_acc = []
"""
-------------------------------------------------
ADDESTRAMENTO CON K-FOLD CROSS-VALIDATION (STUDIO)
-------------------------------------------------
"""
fold = 1
for train_index, val_index in kf.split(X_temp):
    print(f"\n--- 🔄 INIZIO FOLD {fold}/5 ---")

    # Estraiamo i dati per questo specifico giro usando gli indici
    X_train_f, X_val_f = X_temp[train_index], X_temp[val_index]
    y_train_f, y_val_f = y_temp.iloc[train_index].values, y_temp.iloc[val_index].values

    # TRASFORMAZIONE IN TENSORI
    X_train_tensore = torch.from_numpy(X_train_f).float()
    y_train_tensore = torch.from_numpy(y_train_f).float()
    X_val_tensore = torch.from_numpy(X_val_f).float()
    y_val_tensore = torch.from_numpy(y_val_f).float()

    # CREAZIONE DATALOADER
    train_dataset = TensorDataset(X_train_tensore, y_train_tensore)
    val_dataset = TensorDataset(X_val_tensore, y_val_tensore)

    # Shuffle true per evitare che il modello impara bias (distorsione) legati all'ordine dei dati
    # In sintesi costringo il mio modello a capire relazioni tra le features e non in base all'ordine dei dati.
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

    # RE-INIZIALIZZAZIONE MODELLO (FONDAMENTALE PER NON "BARARE"), SCELGO TRA DUE DIVERSI MODELLI (CNN e MLP)
    # Passo al modello le mie features che dopo essere state processate(standard_scalar e onehoteconding)
    # avranno una dimensione maggiore rispetto a quelle del dataset iniziale.
    # Uso .to(device) per passare tutto alla GPU NVIDIA.
    #model = RilevatoreAttacchiCNN(X_processed.shape[1]).to(device)
    model = RilevatoreAttacchi(X_processed.shape[1], 64, 32).to(device)

    # Definizione dell'ottimizzatore (Optimizer) scelgo Adam che modifica il passo del 'lr' in maniera intelligente
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    '''#peso_attacco = torch.tensor([1.5]).to(device)  # Definiamo il peso e lo spostiamo sul device (CPU o GPU) questo per dare più pesi ai falsi negativi
    criterion = nn.BCEWithLogitsLoss(pos_weight=peso_attacco)
    '''

    # Definizione della funzione di perdita, utilizzo una binary cross entropy perchè devo decidere se il mio output è un attacco o no
    # Quindi scelgo tra due possibili valori inoltre ingloba la funzione sigmoide che trasforma il mio logit grezzo in un numero tra 0 e 1, perfetto
    # per una probabilità
    criterion = nn.BCEWithLogitsLoss()

    # Salvo risultati per visualizzazione grafici
    fold_train_losses = []
    fold_val_losses = []
    fold_val_accs = []

    print("\n" + "=" * 50)
    print("--- INIZIO DELLA FASE DI ADDESTRAMENTO E VALUTAZIONE DEL MODELLO ---")
    print("=" * 50)

    num_epoche = 13

    for epoca in range(num_epoche):
        model.train()  # Impostiamo il modello in modalità addestramento
        loss_totale = 0

        for batch_x, batch_y in train_loader:
            # 1. Spostiamo i dati sulla GPU
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 2. Reset dei gradienti: Fondamentale poichè Pytorch accumula i gradienti per ogni batch
            optimizer.zero_grad()

            # 3. Forward pass (previsione)
            outputs = model(batch_x)

            # 4. Calcolo della perdita (Loss)
            # BCEWithLogitsLoss vuole che le etichette siano nella stessa forma (shape) dell'output
            loss = criterion(outputs, batch_y.unsqueeze(1))

            # 5. Backpropagation
            loss.backward()

            # 6. Aggiornamento dei pesi
            optimizer.step()

            loss_totale += loss.item()
            # ... (fine del ciclo for batch_x, batch_y in train_loader)

        # --- FASE DI VALIDAZIONE ---
        model.eval()  # Disattiva layer come il Dropout o la Batch Normalization che servono solo durante lo studio.
        corrette_val = 0  # Contatore per le previsioni azzeccate
        totale_val = 0  # Contatore per il numero totale di campioni
        loss_val_totale = 0
        with torch.no_grad():  # Fondamentale poichè non c'è bisogno di calcolare nessun gradiente, risparmio memoria
            for batch_x_val, batch_y_val in val_loader:
                batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)

                outputs_val = model(batch_x_val)
                loss_v = criterion(outputs_val, batch_y_val.unsqueeze(1))
                loss_val_totale += loss_v.item()

                # --- CALCOLO ACCURACY ---
                # 1. Applichiamo la sigmoide e la soglia a 0.5
                probabilità = torch.sigmoid(outputs_val)
                previsioni = (probabilità > 0.4).float()

                # 2. Confrontiamo con le etichette reali
                # .sum() conta quante sono uguali, .item() lo trasforma in numero Python
                corrette_val += (previsioni == batch_y_val.unsqueeze(1)).sum().item()
                totale_val += batch_y_val.size(0)

            accuracy_val = corrette_val / totale_val

        # Calcoliamo la media delle loss
        loss_addestramento = loss_totale / len(train_loader)
        loss_validazione = loss_val_totale / len(val_loader)

        print(f"Epoca [{epoca + 1}/{num_epoche}]")
        print(f"  - Train Loss: {loss_addestramento:.4f} | Val Loss: {loss_validazione:.4f}")
        print(f"  - Val Accuracy: {accuracy_val:.4%}")  # Mostra l'accuratezza in percentuale

        # Salvo valori per i grafici
        fold_train_losses.append(loss_addestramento)
        fold_val_losses.append(loss_validazione)
        fold_val_accs.append(accuracy_val)

    # Per grafici
    history_train_loss.append(fold_train_losses)
    history_val_loss.append(fold_val_losses)
    history_val_acc.append(fold_val_accs)

    # Salvataggio risultato del fold
    risultati_val_accuracy.append(accuracy_val)
    print(f"✅ Fold {fold} completato. Accuracy: {accuracy_val:.4%}")
    fold += 1

# Calcolo media K-Fold per il report
print(f"\n📊 Accuratezza Media Cross-Validation: {np.mean(risultati_val_accuracy):.2%}")

"""
---------------------------------------
VALUTAZIONE FINALE SUL TEST SET (SCORE MODEL)
---------------------------------------
"""
# Liste per accumulare i risultati
y_true = []
y_pred = []

model.eval()
corrette_test = 0
totale_test = 0

with torch.no_grad():
    for batch_x_test, batch_y_test in test_loader:
        # Spostiamo i dati sulla GPU
        batch_x_test, batch_y_test = batch_x_test.to(device), batch_y_test.to(device)

        # Forward pass (previsione)
        outputs_test = model(batch_x_test)

        # Trasformiamo i logits in classi 0 o 1
        probabilità_test = torch.sigmoid(outputs_test)
        previsioni_test = (probabilità_test > 0.4).float()

        # Conteggio dei successi
        corrette_test += (previsioni_test == batch_y_test.unsqueeze(1)).sum().item()
        totale_test += batch_y_test.size(0)

        # --- AGGIUNTA ALLE LISTE  ---
        # .cpu() riporta i dati dalla GPU alla memoria principale
        # .numpy() e .tolist() servono per Scikit-learn
        y_true.extend(batch_y_test.cpu().numpy().tolist())
        y_pred.extend(previsioni_test.cpu().squeeze().numpy().tolist())

# Risultato finale
accuracy_test = corrette_test / totale_test
print(f"\n🏆 ACCURATEZZA DEFINITIVA SUL TEST SET: {accuracy_test:.2%}")

# Ora stampiamo i risultati per il report
print("\n--- RISULTATI DEFINITIVI (SCORE MODEL) ---")
print(classification_report(y_true, y_pred, target_names=['Normale', 'Attacco']))

print("\n--- MATRICE DI CONFUSIONE ---")
print(confusion_matrix(y_true, y_pred))

print("Addestramento completato! Generazione grafici...")
plotta_risultati_cv(history_val_acc, nome_metrica="Accuracy")
# Grafico per la Loss di Addestramento (in arancione)
plotta_risultati_cv(history_train_loss, nome_metrica="Train Loss", colore="orange")

# Grafico per la Loss di Validazione (in rosso)
plotta_risultati_cv(history_val_loss, nome_metrica="Val Loss", colore="red")

plt.show()

exit()

