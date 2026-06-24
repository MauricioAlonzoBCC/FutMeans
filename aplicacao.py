import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



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


###########################
# # DESCOMENTAR PARTE ABAIXO PARA VIZUALIZAÇÃO EM 2 DIMENSOES

# #  Instanciar o PCA para reduzir os dados para 2 dimensões
# pca = PCA(n_components=2, random_state=42)

# #  Aplicar o PCA nos dados padronizados (X_scaled)
# X_pca = pca.fit_transform(X_scaled)


# df["PC1"] = X_pca[:, 0]
# df["PC2"] = X_pca[:, 1]

# # Extrair a variância explicada (MUITO importante para o artigo)
# var_explicada = pca.explained_variance_ratio_
# print(f"Variância explicada pelo PC1: {var_explicada[0] * 100:.2f}%")
# print(f"Variância explicada pelo PC2: {var_explicada[1] * 100:.2f}%")
# print(f"Variância total preservada: {(var_explicada[0] + var_explicada[1]) * 100:.2f}%")

# #  Criar a visualização gráfica
# plt.figure(figsize=(12, 8))

# # scatterplot do seaborn colore automaticamente os pontos baseados no "Cluster"
# sns.scatterplot(
#     x="PC1", 
#     y="PC2",
#     hue="Cluster",
#     palette="viridis", # Uma paleta de cores muito elegante para artigos
#     data=df,
#     alpha=0.7,         # Transparência para ver sobreposição de pontos
#     edgecolor="k"      # Borda preta nos pontos
# )


# plt.title("Projeção 2D dos Clusters de Jogadores usando PCA", fontsize=14, fontweight='bold')
# plt.xlabel(f"Componente Principal 1 (PC1) - {var_explicada[0] * 100:.1f}% da variância explicada", fontsize=12)
# plt.ylabel(f"Componente Principal 2 (PC2) - {var_explicada[1] * 100:.1f}% da variância explicada", fontsize=12)


# plt.legend(title='Clusters', title_fontsize='13', fontsize='11', loc='best')
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.show()

##########################




###################################################################33
# # DESCOMENAAR ESSA PARTE PARA VISUALIZAR GRAFICO COTOVELO 


# valores_j = []

# # Testando K de 1 a 10 grupos
# intervalo_k = range(1, 11)

# for k in intervalo_k:
#     # Rodando o modelo para cada valor de K
#     modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     modelo_kmeans.fit(X_scaled)
    
#     # Extraindo a métrica J (Inércia)
#     valores_j.append(modelo_kmeans.inertia_)


# plt.figure(figsize=(10, 6))
# plt.plot(intervalo_k, valores_j, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)


# plt.title("Método do Cotovelo: Minimização da Função Objetivo J", fontsize=14, fontweight='bold')
# plt.xlabel("Número de Clusters (k)", fontsize=12)
# plt.ylabel("Inércia (J)", fontsize=12)
# plt.xticks(intervalo_k)
# plt.grid(True, linestyle='--', alpha=0.7)


# plt.show()
############################################################