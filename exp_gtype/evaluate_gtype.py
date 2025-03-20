import pickle
import random
from tabulate import tabulate

random.seed(10)

cents = ["betweenness", "closeness", "page_rank", "clustering"]

def read_data(centrality, layers):
    data = []
    for i in range(1, layers + 1):
        with open(centrality + f"/{i}_GNN_kt.pickle", "rb") as f:
            arrays = pickle.load(f)
            data.append(arrays[0][0])
    return data

dicc = {}

for cent in cents:
    data = read_data("results/" + cent, 12)
    dicc[cent] = data

# Crear una tabla con tabulate
headers = ["Número de capas intermedias"] + cents
rows = [[i + 1] + [dicc[cent][i] for cent in cents] for i in range(12)]

# Mostrar la tabla
print(tabulate(rows, headers=headers, tablefmt="grid"))
