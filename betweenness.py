 #------ 0º INICIALIZACIÓN ---------------------
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

#---------------------------EXPERIMENTACIÓN------------------------
'''

1º Se extrae del comando del archivo el tipo de grafos sintéticos a utilizar. Esto servirá para
    MIRAR LOS DATASETS ADECUADOS. Siempre miraran los datasets de datasets/data_splits

2º Se fija el tamaño del modelo, el cuál será igual al tamaño del grafo más grande (el max_nodes).
    Después, de los datasets de entrenamiento y de tests, se extrae la lista de matrices de adjacencia
    y la lista de matrices de adjacencia transpuesta. MÁS DETALLES DENTRO DE LA FUNCIÓN

3º Una vez extraido la lista de matrices de adyacencia (tanto la normal como la transpuesta), lo que
    se hace es CREAR EL MODELO A ENTRENAR. En este se configura:
        - la cantidad de capas intermedias
        - El optimizador y el learning_rate del mismo.
        - El número de épocas
        - La localización de la máquina desde la cuál se ejecutará el código(GPU= Por memoria)

4º  Una vez configurado el modelo, por cada época se realiza el entrenamiento y la prueba del modelo.
    MÁS INFORMACIÓN EN SUS FUNCIONES

'''
#-------------------AQUÍ OCURRE EL 1º PASO-----------------------
if gtype == "SF":
    data_path = "./datasets/data_splits/SF/betweenness/"
    print("Scale-free graphs selected.")

elif gtype == "ER":
    data_path = "./datasets/data_splits/ER/betweenness/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP/betweenness/"
    print("Gaussian Random Partition graphs selected.")

# Lo que se hace es preparar los paquetes a utilizar, fijar una semilla PARA LAS OPERACIONES 
# DE TORCH, y leer e interpretar los argumentos del comando de entrada.
#Para esta lectura, se lee el parámetro "g", que sirve para indicar que tipo de datos sintéticos
# a utilizar. Estan los grafos Gaussianos, Erdo-Renyi y Scale-free. Todos los grafos usados son
# dirigidos.



#Load training data
print(f"Loading data...")
with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)


with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)


#-------------------------------------------------------------------

model_size = 10000 # MAX_NODES DEL GENERATE_GRAPH
#Una vez abierto los grafos, obtiene las matrices de adyacencia de los mismos.
print(f"Graphs to adjacency conversion.")

list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)


def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    '''
    Se va recorriendo cada elemento del dataset de entrenamiento de la siguiente forma:
    
    1º Extrae la matriz de adyacencia normal y transpuesta del elemento y se lo mete al modelo. Esto
        le devuelve la propia salida, que será un array numpy con las centralidades ESTIMADAS
    2º Extrae de la lista de matrices de centralidad la matriz de centralidad del elemento. Con
        esto, se compara con lo estimado y se guarda su error. OJO, PARA CALCULAR EL ERROR,
        ESCOGEN 20*N_i pares de nodos posibles.
    
    3º Con ese error + el error de entrenamiento acumulado, hace BACKPROPAGATION para actualizar
        los parámetros según el optimizador. Una vez hecho esto, con el siguiente elemento antes
        se resetean los gradientes del optimizador.
    '''
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
    '''
    Se va recorriendo cada elemento del dataset de entrenamiento de la siguiente forma:
    
    1º Extrae la matriz de adyacencia normal y transpuesta del elemento y se lo mete al modelo. Esto
        le devuelve la propia salida, que será un array numpy con las centralidades ESTIMADAS
    2º Extrae de la lista de matrices de centralidad la matriz de centralidad del elemento. 
    
    3º AHORA, la diferencia esta en que se compara lo estimado con lo real para clasificar
        la correlación. Para el coeficiente de KT, se tienen en cuenta todos los pares de nodos
    '''
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
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)
    
        kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
        list_kt.append(kt)
        #g_tmp = list_graph_test[j]
        #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")

    print(f"   Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")


#-----------------AQUI OCURRE EL PASO 3º------------------------------------
#Model parameters
hidden = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
#RECUERDA, los optimizadores se encargaban de la propia actualización
# de los parámetros.
num_epoch = 15
#----------------------------------------------------------------------
#-----------------AQUÍ OCURRE EL PASO 4º---------------------------------
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

#-------------------------------------------------------------------------

    