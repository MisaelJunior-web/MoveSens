import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier   
from sklearn.metrics import classification_report, accuracy_score

arquivo_gestos = {
    "joia":0,
    "paz":1,
    "negativo":2,
    "hangloose":3,
    "rock":4,
    "coracao":5
}

dados=[]

for nome, label in arquivo_gestos.items():
    df = pd.read_csv(Path("dados") / f"{nome}.csv", header=None)

    if df.shape[1]!=42:
        raise ValueError(
            f"{nome}.csv tem {df.shape[1]} colunas;"
            "eram esperadas 42."
        )

    df["label"] = label
    dados.append(df)

data = pd.concat(dados, ignore_index=True)
data = data.dropna()

X = data.drop(columns=["label"])
Y = data["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)

model.fit(X_train, Y_train)

predicoes = model.predict(X_test)

print("Acurácia:", accuracy_score(Y_test, predicoes))
print("\nRelatório por gesto:")
print(
    classification_report(
        Y_test,
        predicoes,
        labels=list(arquivo_gestos.values()),
        target_names=list(arquivo_gestos.keys()),
        zero_division=0
    )
)

joblib.dump(model, "modelo_gestos.pkl")

print("\nModelo salvo em modelo_gestos.pkl")