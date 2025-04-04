 #------ 0º INICIALIZACIÓN ---------------------
import numpy as np
import pickle
import networkx as nx
import torch
from exp_layer_type.conv_betweenness import CNN_Bet
from exp_layer_type.gat_betweenness import GAT_Bet
from exp_layer_type.sage_betweenness import GSAGE_Bet
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet
import argparse

#Loading graph data
parser = argparse.ArgumentParser()
parser.add_argument("--g",default="SF")
parser.add_argument("--num_intermediate_layer",type=int,default=6)
parser.add_argument("--gnn",default="GNN")
parser.add_argument("--model_size",type=int,default=10000)
parser.add_argument("--version",default="")
parser.add_argument("--g_hype",type=float,default=None)
parser.add_argument("--optional_name",type=str,default="")
args = parser.parse_args()
gtype = args.g
num=args.num_intermediate_layer
gnn_type=args.gnn
model_size=args.model_size
v=args.version
g_hype=args.g_hype
optional=args.optional_name
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

elif gtype == "TU":
    data_path = "./datasets/data_splits/TU/betweenness/"
    print("Turan graphs selected.")

elif gtype == "FT":
    data_path = "./datasets/data_splits/FT/betweenness/"
    print("Full Rary Tree graphs selected.")

elif gtype == "FOR_EXP":
    data_path = "./datasets/data_splits/FOR_EXP/betweenness/"
    print("Real data experimentation")
    
elif gtype == "HYP":
    data_path = "./datasets/data_splits/HYP/betweenness/"
    print("Real data experimentation")

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


#Una vez abierto los grafos, obtiene las matrices de adyacencia de los mismos.
print(f"Graphs to adjacency conversion.")

#Recuerda, estos son listas de matrices de adyacencia. Las matrices tienen model_size columnas.
list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)


def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    
    with torch.autograd.set_detect_anomaly(True):
        '''
        Se va recorriendo cada elemento del dataset de entrenamiento de la siguiente forma:
        
        1º Extrae la matriz de adyacencia normal y transpuesta del elemento y se lo mete al modelo. Esto
            le devuelve la propia salida, que será un array numpy con las centralidades ESTIMADAS
        2º Extrae de la lista de matrices de centralidad la matriz de centralidad del elemento. Con
            esto, se compara con lo estimado y se guarda su error. OJO, PARA CALCULAR EL ERROR,
            ESCOGEN 20*N_i pares de nodos posibles.
        
        3º Con ese error, hace BACKPROPAGATION para actualizar
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
            #RECUERDA, y_out=Array, bc_mat_train[:,i]= Array.
            true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
            true_val = true_arr.to(device)
            
            loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
            loss_train = loss_train + float(loss_rank) #Esto realmente no sirve
    #        torch.autograd.set_detect_anomaly(True)

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
    return np.mean(np.array(list_kt)),np.std(np.array(list_kt))


#-----------------AQUI OCURRE EL PASO 3º------------------------------------
#Model parameters
hidden = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=None
if gnn_type=="GNN":
    model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6,num_intermediate_layers=num)
elif gnn_type=="CNN":
    model = CNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6,num_intermediate_layers=num)
elif gnn_type=="GAT":
    model = GAT_Bet(ninput=model_size,nhid=hidden,dropout=0.6,num_intermediate_layers=num)
elif gnn_type=="SAGE":
    model = GSAGE_Bet(ninput=model_size,nhid=hidden,dropout=0.6,num_intermediate_layers=num)


model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
#RECUERDA, los optimizadores se encargaban de la propia actualización
# de los parámetros.
num_epoch = 15
#----------------------------------------------------------------------
#-----------------AQUÍ OCURRE EL PASO 4º---------------------------------
print("Training")
print(f"Total Number of epoches: {num_epoch}")
kt_mean=None
std_kt=None
list_data_per_epoch=list()
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)

    #to check test loss while training
    with torch.no_grad():
        kt_mean,std_kt=test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)
    
    list_data_per_epoch.append([kt_mean,std_kt,model.get_num_intermediate_layers(),model.get_gnn_type()])
    
with open(f"results/betweenness/{model.get_num_intermediate_layers()}_{model.get_gnn_type()}_{gtype}_per_epoch_{v}_kt.pickle","wb") as fopen2:
    pickle.dump(list_data_per_epoch,fopen2)
print("")
print("Results saved")
#-------------------------------------------------------------------------
#Código para guardar resultados del KT obtenido EN LA ÚLTIMA ÉPOCA
if v!="":
    v=f"_{v}"
    
if optional!="":
    optional=f"_{optional}"
list_data=list()
list_data.append([kt_mean,std_kt,model.get_num_intermediate_layers(),model.get_gnn_type(),g_hype])
with open(f"results/betweenness/{model.get_num_intermediate_layers()}_{model.get_gnn_type()}_{gtype}{optional}{v}_kt.pickle","wb") as fopen2:
        pickle.dump(list_data,fopen2)
print("")
print("Results saved")
