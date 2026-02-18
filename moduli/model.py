import torch.nn as nn
import torch.nn.functional as F

class RilevatoreAttacchi(nn.Module):     # Ereditiamo da nn.Module
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(RilevatoreAttacchi, self).__init__()

        # Strato nascosto 1 (prende le feature di X_processed)
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)

        # Strato nascosto 2
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)

        # Strato di output (restituisce il logit)
        self.output = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        # Restituiamo il logit puro per BCEWithLogitsLoss
        return self.output(x)


class RilevatoreAttacchiCNN(nn.Module):
    def __init__(self, input_dim):
        super(RilevatoreAttacchiCNN, self).__init__()

        # STRATO CONVOLUZIONALE 1D
        # in_channels=1 (il canale)
        # out_channels=64 (il numero di filtri)
        # kernel_size=17 (la dimensione della finestra)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=17)

        #BATCH NORMALIZATION primo strato
        self.bn1 = nn.BatchNorm1d(64)

        #SECONDO STRATO CONVOLUZIONALE
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)

        #BATCH NORMALIZATION secondo strato
        self.bn2 = nn.BatchNorm1d(128)

        # STRATO DI POOLING (Max Pooling)
        # Prende il valore più alto
        # Questo è un trucco molto usato: invece di dover calcolare esattamente quanti numeri escono dalla convoluzione,
        # questo strato prende il valore più alto filtrato.
        self.gap = nn.AdaptiveMaxPool1d(1)

        # STRATO DI DROPOUT per spegnere neuroni e rendere il modello più robusto
        self.drop = nn.Dropout(0.5)

        # STRATO FINALE (Fully Connected)
        # Riceve 64 valori (uno per ogni filtro) e produce 1 output (il logit)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # RESHAPE: trasformiamo (Batch, Features) -> (Batch, 1, Features), sostanzialmente aggiungo una dimensione al mio x_processed essendo che le CNN
        # si aspettano 3 dimensioni
        x = x.unsqueeze(1)

        # Applichiamo la convoluzione
        x = self.conv1(x)

        # Normalizzare i dati prima di passarla alla ReLu
        x = self.bn1(x)

        # Attivazione ReLU 1 strato
        x = F.relu(x)

        # Convoluzione secondo strato
        x = self.conv2(x)

        # Normalizzare i dati prima di passarla alla ReLu del secondo strato
        x = self.bn2(x)

        # Attivazione ReLU 2 strato
        x = F.relu(x)

        # Max Pooling
        x = self.gap(x)

        # Dropout
        x = self.drop(x)

        # Flatten: trasformiamo da (Batch, 64, 1) -> (Batch, 64)
        x = x.view(x.size(0), -1)

        # Classificazione finale
        x = self.fc(x)
        return x