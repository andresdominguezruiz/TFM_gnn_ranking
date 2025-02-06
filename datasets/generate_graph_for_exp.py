import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
import argparse
import os

np.random.seed(1)


# Función para crear grafos de tipo Scale-Free
def create_graph(num_nodes):
    alpha = np.random.randint(40, 60) * 0.01
    gamma = 0.05
    beta = 1 - alpha - gamma
    g_nx = nx.scale_free_graph(num_nodes, alpha=alpha, beta=beta, gamma=gamma)
    return g_nx


# Convierte NetworkX a NetworkIt
def nx2nkit(g_nx):
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)

    for i in range(node_num):
        g_nkit.addNode()

    for e1, e2 in g_nx.edges():
        g_nkit.addEdge(e1, e2)

    return g_nkit


# Calcula la centralidad de intermediación (betweenness)
def cal_exact_bet(g_nx):
    g_nkit = nx2nkit(g_nx)  # Se debe convertir antes de usar NetworkIt
    exact_bet = centrality.Betweenness(g_nkit, normalized=True).run().ranking()
    exact_bet_dict = {j[0]: j[1] for j in exact_bet}
    return exact_bet_dict


# Calcula la centralidad de cercanía (closeness)
def cal_exact_close(g_nx):
    g_nkit = nx2nkit(g_nx)  # Se debe convertir antes de usar NetworkIt
    exact_close = centrality.Closeness(g_nkit, True, 1).run().ranking()
    exact_close_dict = {j[0]: j[1] for j in exact_close}
    return exact_close_dict


# Configuración de argumentos desde la terminal
parser = argparse.ArgumentParser(description="Generar grafos y calcular centralidades.")
parser.add_argument("--num_graphs", type=int, required=True, help="Número de grafos a generar")
parser.add_argument("--num_nodes", type=int, required=True, help="Número de nodos por grafo")

args = parser.parse_args()
num_of_graphs = args.num_graphs
num_nodes = args.num_nodes

graph_types = ["SF"]

# Crear carpeta de salida si no existe
os.makedirs("./graphs", exist_ok=True)

for graph_type in graph_types:
    print("###################")
    print(f"Generating graph type : {graph_type}")
    print(f"Number of graphs to be generated: {num_of_graphs}")
    list_bet_data = []
    list_close_data = []
    print("Generating graphs and calculating centralities...")

    for i in range(num_of_graphs):
        print(f"Graph index: {i+1}/{num_of_graphs}", end='\r')
        g_nx = create_graph(num_nodes)  # Usa el número de nodos proporcionado por el usuario
        
        # Eliminar nodos aislados y reindexar si es necesario
        if nx.number_of_isolates(g_nx) > 0:
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)

        bet_dict = cal_exact_bet(g_nx)
        close_dict = cal_exact_close(g_nx)

        list_bet_data.append([g_nx, bet_dict])
        list_close_data.append([g_nx, close_dict])

    name = "FOR_EXP"
    fname_bet = f"./graphs/{name}_data_bet.pickle"
    fname_close = f"./graphs/{name}_data_close.pickle"

    with open(fname_bet, "wb") as fopen:
        pickle.dump(list_bet_data, fopen)

    with open(fname_close, "wb") as fopen1:
        pickle.dump(list_close_data, fopen1)

    print("\nGraphs saved")

print("End.")
