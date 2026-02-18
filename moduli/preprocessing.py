import numpy as np



def split_features(df):
#Dall'eplorazione del dataset mi sono accorto che molte features sono --> Dtype = Object.
#Questo non va bene perchè il modello elaborerà solo formati di tipo numerico (int, float, etc)
#Quindi utilizzerò "StandardScaler" per le features numeriche e "OneHotEncoder" per quelle testuali(objet) per renderle numeriche.

    all_numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    all_object_features = df.select_dtypes(include='object').columns.tolist()
    print(f"\nTutte le features numeriche sono:{all_numeric_features}")
    print(f"\nTutte le features categoriche sono:{all_object_features}")

    numeric_features = []
    object_features = []

    #Feature target (y) quindi da droppare
    for col in all_numeric_features:
        if col != "attack_detected":
            numeric_features.append(col)

#Il modello potrebbe imparare a memoria gli specifici ID di sessione invece di estrarre pattern utili dalle altre features
#Quindi elimino dalla features anche "session_id"
    for col in all_object_features:
        if col != "session_id":
            object_features.append(col)

    print("\nLe features numeriche sono",numeric_features)
    print("\nLe features categoriche sono",object_features)

    return numeric_features, object_features
