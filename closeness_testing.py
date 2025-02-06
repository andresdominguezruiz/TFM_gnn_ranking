import numpy as np
import pickle
import networkx as nx
import torch
import subprocess
import os
import random
import torch.nn as nn
import argparse
from model_close import GNN_Close
from utils import *
from model_bet import GNN_Bet

torch.manual_seed(20)

# **1. Configurar los argumentos desde la terminal**
parser = argparse.ArgumentParser(description="Entrenar modelo con grafos generados y probarlo con un grafo real.")
parser.add_argument("--file", type=str, required=True, help="Archivo .txt del grafo de prueba")
parser.add_argument("--num_permutations", type=int, required=True, help="Número de permutaciones a realizar")

args = parser.parse_args()
test_graph_path = args.file
num_permutations = args.num_permutations

# **2. Cargar el grafo de prueba desde el archivo .txt**
def load_txt_graph(file_path):
    G = nx.DiGraph()

    with open(file_path, "r") as f:
        edges = set()
        nodes = set()
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            node1, node2 = map(int, line.split())
            edges.add((node1, node2))
            nodes.update([node1, node2])

    G.add_edges_from(edges)
    G.add_nodes_from(nodes)

    mapping = {old: new for new, old in enumerate(G.nodes())}
    G_reset = nx.relabel_nodes(G, mapping)

    return G_reset, list(G_reset.nodes())

# **3. Cargar el grafo de prueba y obtener su tamaño**
print(f"Loading test graph from {test_graph_path}...")
test_graph, test_node_sequence = load_txt_graph(test_graph_path)
num_nodes_test = test_graph.number_of_nodes()
print(f"Test graph has {num_nodes_test} nodes.")

# **4. Ejecutar el script anterior para generar grafos de entrenamiento**
print("Generating training graphs with the same number of nodes as the test graph...")
subprocess.run(["python", "datasets/generate_graph_for_exp.py", "--num_graphs", "5", "--num_nodes", str(num_nodes_test)])

# **5. Cargar los grafos de entrenamiento generados**
data_path = "./graphs/FOR_EXP_data_close.pickle"
print(f"Loading training graphs from {data_path}...")
with open(data_path, "rb") as fopen:
    list_graph_train, close_mat_train = zip(*pickle.load(fopen))

# **6. Convertir datos de entrenamiento**
close_mat_train = np.array([
    [bc.get(node, 0) for node in g.nodes()]
    for g, bc in zip(list_graph_train, close_mat_train)
]).T  

list_n_seq_train = [list(g.nodes()) for g in list_graph_train]
list_num_node_train = [g.number_of_nodes() for g in list_graph_train]

# **7. Generar permutaciones**
print(f"Generating {num_permutations} adjacency matrices with node sequence permutations...")
list_graph_train_permuted = []
list_n_seq_train_permuted = []
list_num_node_train_permuted = []
close_mat_permuted = []

for g, seq, bc in zip(list_graph_train, list_n_seq_train, close_mat_train.T):
    for _ in range(num_permutations):
        permuted_seq = random.sample(seq, len(seq))
        list_graph_train_permuted.append(g)
        list_n_seq_train_permuted.append(permuted_seq)
        list_num_node_train_permuted.append(g.number_of_nodes())
        close_mat_permuted.append(bc)

close_mat_train = np.array(close_mat_permuted).T  

# **8. Preparar el grafo de prueba**
list_graph_test = [test_graph]
list_n_seq_test = [test_node_sequence]
list_num_node_test = [test_graph.number_of_nodes()]

# **9. Calcular la centralidad de intermediación**
close_mat_test = np.zeros((test_graph.number_of_nodes(), 1))
closeness = nx.closeness_centrality(test_graph)
for node, centrality in closeness.items():
    close_mat_test[node, 0] = centrality

# **10. Convertir grafos a matrices de adyacencia**
model_size = num_nodes_test
print(f"Converting graphs to adjacency matrices...")
list_adj_train, list_adj_t_train = graph_to_adj_close(list_graph_train_permuted, list_n_seq_train_permuted, list_num_node_train_permuted, model_size)
list_adj_test, list_adj_t_test = graph_to_adj_close(list_graph_test, list_n_seq_test, list_num_node_test, model_size)

# **11. Definir funciones de entrenamiento y prueba**
def train(list_adj_train, list_adj_t_train, list_num_node_train, close_mat_train):
    model.train()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    
    for i in range(num_samples_train):
        adj = list_adj_train[i].to(device)
        adj_t = list_adj_t_train[i].to(device)

        optimizer.zero_grad()
        y_out = model(adj, adj_t)
        
        true_arr = torch.from_numpy(close_mat_train[:, i]).float().to(device)
        loss_rank = loss_cal(y_out, true_arr, list_num_node_train[i], device, model_size)

        loss_train += float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test, list_adj_t_test, list_num_node_test, close_mat_test):
    model.eval()
    list_kt = []
    num_samples_test = len(list_adj_test)

    for j in range(num_samples_test):
        adj = list_adj_test[j].to(device)
        adj_t = list_adj_t_test[j].to(device)

        y_out = model(adj, adj_t)
        true_arr = torch.from_numpy(close_mat_test[:, j]).float().to(device)
        kt = ranking_correlation(y_out, true_arr, list_num_node_test[j], model_size)

        list_kt.append(kt)

    print(f"Average KT score on test graph: {np.mean(np.array(list_kt))}")

# **12. Inicializar modelo y optimizador**
hidden = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Close(ninput=model_size, nhid=hidden, dropout=0.6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
num_epoch = 15

# **13. Entrenar el modelo**
print("Training model...")
for e in range(num_epoch):
    print(f"Epoch {e+1}/{num_epoch}")
    train(list_adj_train, list_adj_t_train, list_num_node_train_permuted, close_mat_train)
 
    with torch.no_grad():
        test(list_adj_test, list_adj_t_test, list_num_node_test, close_mat_test)

print("Training complete.")
