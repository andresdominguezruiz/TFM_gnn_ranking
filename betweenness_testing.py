import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet
import argparse

torch.manual_seed(20)

# Definir argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--g", default="SF")
args = parser.parse_args()
gtype = args.g
print(gtype)

# **1. Cargar datos de entrenamiento desde FOR_EXP_data_bet.pickle**
print(f"Loading graphs from FOR_EXP_data_bet.pickle...")

data_path = "./datasets/graphs/FOR_EXP_data_bet.pickle"

with open(data_path, "rb") as fopen:
    list_graph_train, bc_mat_train = zip(*pickle.load(fopen))  # Cargar grafos y betweenness centrality

# Convertir diccionarios de betweenness centrality en una matriz NumPy
bc_mat_train = np.array([
    [bc.get(node, 0) for node in g.nodes()]  # Obtener valores en orden de los nodos del grafo
    for g, bc in zip(list_graph_train, bc_mat_train)
]).T  # Transponer para mantener el formato esperado

# Obtener la secuencia de nodos de cada grafo
list_n_seq_train = [list(g.nodes()) for g in list_graph_train]

# **2. Generar 200 matrices de adyacencia con permutaciones de la secuencia de nodos**
print(f"Generating 200 adjacency matrices with node sequence permutations...")

num_permutations = 3 #Al final, el nº de grafos de entrada con esto seria 5 x este número
list_graph_permuted = []
list_n_seq_permuted = []
list_num_node_permuted = []
bc_mat_permuted = []

for g, seq, bc in zip(list_graph_train, list_n_seq_train, bc_mat_train.T):
    for _ in range(num_permutations):
        permuted_seq = random.sample(seq, len(seq))  # Permutar la secuencia de nodos
        list_graph_permuted.append(g)
        list_n_seq_permuted.append(permuted_seq)
        list_num_node_permuted.append(g.number_of_nodes())
        bc_mat_permuted.append(bc)  # Mantener la misma matriz de centralidad

bc_mat_train = np.array(bc_mat_permuted).T  # Ajustar formato de la matriz de centralidad

# **3. Cargar el grafo de prueba desde el archivo WikiTalk.txt**


def load_txt_graph(file_path, num_nodes):
    G = nx.DiGraph()

    # Leer archivo y extraer aristas
    with open(file_path, "r") as f:
        edges = []
        nodes = set()

        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue  # Ignorar comentarios y líneas vacías

            node1, node2 = map(int, line.split())
            edges.append((node1, node2))
            nodes.update([node1, node2])

    # Verificar si hay suficientes nodos
    if len(nodes) < num_nodes:
        raise ValueError(f"El grafo solo tiene {len(nodes)} nodos, pero se solicitaron {num_nodes}.")

    # Seleccionar `num_nodes` nodos aleatorios
    selected_nodes = set(random.sample(nodes, num_nodes))

    # Crear un subgrafo SOLO con los nodos seleccionados y sus aristas
    G_sub = nx.DiGraph()
    for node1, node2 in edges:
        if node1 in selected_nodes and node2 in selected_nodes:
            G_sub.add_edge(node1, node2)

    # 🔹 Reiniciar índices de nodos (0 a `num_nodes-1`)
    mapping = {old: new for new, old in enumerate(G_sub.nodes())}
    G_reset = nx.relabel_nodes(G_sub, mapping)

    # 🔹 Extraer secuencia de nodos asegurando que estén en el grafo
    node_sequence = list(G_reset.nodes())

    return G_reset, node_sequence  # Grafo con nodos reindexados y su secuencia




test_graph_path = "./datasets/real_data/WikiTalk.txt"
test_graph,test_node_sequency = load_txt_graph(test_graph_path,100000)
print(test_graph.number_of_nodes())

# Convertir el grafo de prueba en el formato requerido por el modelo
list_graph_test = [test_graph]
list_n_seq_test = [test_node_sequency]  # Secuencia de nodos
list_num_node_test = [test_graph.number_of_nodes()]
print(test_graph.number_of_nodes())
bc_mat_test = np.zeros((test_graph.number_of_nodes(), 1))  # Matriz de centralidad ficticia si no se tiene


#OBTENCIÓN DEL model_size:
#Según el documento, model_size= max(mayor tamaño de train, mayor tamaño de test)

model_size = max(max(list_num_node_test), max(list_num_node_permuted))


#----------------------------

# **Conversión a matriz de adyacencia**

print(f"Graphs to adjacency conversion.")

list_adj_train, list_adj_t_train = graph_to_adj_bet(
    list_graph_permuted, list_n_seq_permuted, list_num_node_permuted, model_size
)
list_adj_test, list_adj_t_test = graph_to_adj_bet(
    list_graph_test, list_n_seq_test, list_num_node_test, model_size
)

# **Funciones de entrenamiento y prueba**
def train(list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train):
    model.train()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    
    for i in range(num_samples_train):  # num_samples = número de grafos
        adj = list_adj_train[i].to(device)
        adj_t = list_adj_t_train[i].to(device)

        optimizer.zero_grad()
            
        y_out = model(adj, adj_t)
        true_val = torch.from_numpy(bc_mat_train[:, i]).float().to(device)
        
        loss_rank = loss_cal(y_out, true_val, list_num_node_train[i], device, model_size)
        loss_train += float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test):
    model.eval()
    list_kt = []
    
    for j in range(len(list_adj_test)):  
        adj = list_adj_test[j].to(device)
        adj_t = list_adj_t_test[j].to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj, adj_t)
        true_val = torch.from_numpy(bc_mat_test[:, j]).float().to(device)
    
        kt = ranking_correlation(y_out, true_val, num_nodes, model_size)
        list_kt.append(kt)

    print(f"   Average KT score on test graphs is: {np.mean(list_kt)} and std: {np.std(list_kt)}")

# **Configuración del modelo**
hidden = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Bet(ninput=model_size, nhid=hidden, dropout=0.6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epoch = 15

# **Entrenamiento y prueba**
print("Training")
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train, list_adj_t_train, list_num_node_permuted, bc_mat_train)
    with torch.no_grad():
        test(list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test)
