import pickle
import random
from tabulate import tabulate

random.seed(10)

cents = ["betweenness", "closeness", "page_rank", "clustering"]
gtypes=["SF"]
layer=5
model_type="GNN"

def read_data(centrality, types):
    data = []
    for i in types:
        with open(centrality + f"/{layer}_{model_type}_{i}_kt.pickle", "rb") as f:
            arrays = pickle.load(f)
            data.append(arrays[0][0])
    return data

dicc = {}

for cent in cents:
    data = read_data("results/" + cent, gtypes)
    dicc[cent] = data

# Crear una tabla con tabulate
headers = ["Tipos de grafo"] + cents
rows = [[gtypes[i]] + [dicc[cent][i] for cent in cents] for i in range(len(gtypes))]

# Mostrar la tabla
print(f"#############RESULTADOS CON MODELO {model_type} Y {layer} CAPAS INTERMEDIAS######")
print(tabulate(rows, headers=headers, tablefmt="grid"))
