import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("planilha_jogadores.csv")

# Remove jogadores com menos de 300 minutos
df = df[df["Min"] >= 300]

# Preenche valores faltantes de Succ% com 0
df["Succ%"] = df["Succ%"].fillna(0)

features = [
    "Gls",  
    "Ast",   
    "xG",    
    "xAG",   
    "PrgC",  
    "PrgP",
    "PrgR",
    "Sh",
    "SoT",
    "KP",
    "PPA",
    "Tkl",
    "TklW",
    "Int",
    "Tkl+Int",
    "Clr",
    "Touches",
    "Succ",
    "Carries",
    "Rec",
    "Fls",
    "Cmp%",
    "Succ%"
]

for col in features:
    if col not in ["Cmp%", "Succ%"]:
        df[col] = df[col] / df["90s"]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

print(df[["Player", "Pos", "Cluster"]].head(20))

df_pos = df.copy()

# transforma "MF,FW" em ["MF", "FW"]
df_pos["Pos"] = df_pos["Pos"].str.split(",")

# cria uma linha para cada posição
df_pos = df_pos.explode("Pos")

for cluster, grupo in df_pos.groupby("Cluster"):
    print(f"\nCluster {cluster}")
    print(grupo["Pos"].value_counts().to_string())

print()

print("JOGADORES DE EXEMPLO: ")

jogadores = [
    "Ederson",
    "Endrick",
    "Raphinha",
    "Marquinhos",
    "Casemiro",
]

print(
    df[df["Player"].isin(jogadores)]
    [["Player", "Pos", "Squad", "Cluster"]]
)