 
import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet
torch.manual_seed(20)
import argparse

#Loading graph data
parser = argparse.ArgumentParser()
parser.add_argument("--g",default="SF")
args = parser.parse_args()
gtype = args.g
print(gtype)
#La etiqueta que se le pone al final del comando sirve para determinar el tipo de grafos a utilizar

data_path = "./datasets/graphs/FOR_EXP_data_bet.pickle"

#Load training data
print(f"Loading data...")
with open(data_path,"rb") as fopen:
    list_graph_train, bc_mat_train = zip(*pickle.load(fopen)) 

# Convertir diccionarios de betweenness centrality en una matriz NumPy
bc_mat_train = np.array([
    [bc.get(node, 0) for node in g.nodes()]  # Obtener valores en orden de los nodos del grafo
    for g, bc in zip(list_graph_train, bc_mat_train)
]).T  # Transponer para mantener el formato esperado

# Obtener la secuencia de nodos de cada grafo
list_n_seq_train = [list(g.nodes()) for g in list_graph_train]
# Obtener el número de nodos de cada grafo de entrenamiento
list_num_node_train = [g.number_of_nodes() for g in list_graph_train]


def load_txt_graph(file_path):
    G = nx.DiGraph()

    # Leer archivo y extraer aristas
    with open(file_path, "r") as f:
        edges = set()  # Usamos un conjunto para evitar duplicados
        nodes = set()

        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue  # Ignorar comentarios y líneas vacías

            node1, node2 = map(int, line.split())
            edges.add((node1, node2))
            nodes.update([node1, node2])

    # Construir el grafo con todas las aristas extraídas
    G.add_edges_from(edges)

    # Asegurar que todos los nodos están en el grafo (en caso de nodos aislados)
    G.add_nodes_from(nodes)

    # 🔹 Reiniciar índices de nodos (0 a `len(G.nodes) - 1`)
    mapping = {old: new for new, old in enumerate(G.nodes())}
    G_reset = nx.relabel_nodes(G, mapping)

    # 🔹 Extraer secuencia de nodos
    node_sequence = list(G_reset.nodes())

    return G_reset, node_sequence  # Grafo con nodos reindexados y su secuencia


#WikiVote tiene menos nodos
test_graph_path = "./datasets/real_data/Wiki-Vote.txt"
test_graph,test_node_sequency = load_txt_graph(test_graph_path)
print(test_graph.number_of_nodes())

# Convertir el grafo de prueba en el formato requerido por el modelo
list_graph_test = [test_graph]
list_n_seq_test = [test_node_sequency]  # Secuencia de nodos
list_num_node_test = [test_graph.number_of_nodes()]
print(test_graph.number_of_nodes())

# Crear matriz para almacenar los coeficientes de centralidad de intermediación
bc_mat_test = np.zeros((test_graph.number_of_nodes(), 1))

# 🔹 Calcular la centralidad de intermediación
betweenness = nx.betweenness_centrality(test_graph)

# 🔹 Llenar bc_mat_test con los valores de centralidad
for node, centrality in betweenness.items():
    bc_mat_test[node, 0] = centrality
#ES NECESARIO CALCULAR LA CENTRALIDAD REAL DE LOS NODOS PARA REALIZAR LAEVALUACIÓN CORRECTAMENTE




model_size = 7115
#Una vez abierto los grafos, obtiene las matrices de adyacencia de los mismos.
print(f"Graphs to adjacency conversion.")

list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)





def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train): #num_samples= numero de grafos
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):  
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)
        
        print("BC_MAT_TEST------------")
        print(bc_mat_test[:,j])
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)
        
        print("Y_OUT---------------")
        print(y_out)
        print("TRUE_VAL---------------")
        print(true_val)
    
        kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
        list_kt.append(kt)
        #g_tmp = list_graph_test[j]
        #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")

    print(f"   Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")



#Model parameters
hidden = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
num_epoch = 15

print("Training")
print(f"Total Number of epoches: {num_epoch}")
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)

    #to check test loss while training
    with torch.no_grad():
        test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)
#test on 10 test graphs and print average KT Score and its stanard deviation
#with torch.no_grad():
#    test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)


    