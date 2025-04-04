 
import numpy as np
import pickle
import networkx as nx
import torch
from exp_layer_type.conv_clustering import CNN_Clustering
from exp_layer_type.sage_clustering import GSAGE_Clustering
from exp_layer_type.gat_clustering import GAT_Clustering
from utils import *
import random
import torch.nn as nn
from model_clustering import GNN_Clustering

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
if gtype == "SF":
    data_path = "./datasets/data_splits/SF/clustering/"
    print("Scale-free graphs selected.")

elif gtype == "ER":
    data_path = "./datasets/data_splits/ER/clustering/"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP/clustering/"
    print("Gaussian Random Partition graphs selected.")

elif gtype == "TU":
    data_path = "./datasets/data_splits/TU/clustering/"
    print("Turan graphs selected.")

elif gtype == "FT":
    data_path = "./datasets/data_splits/FT/clustering/"
    print("Full Rary Tree graphs selected.")

elif gtype == "FOR_EXP":
    data_path = "./datasets/data_splits/FOR_EXP/clustering/"
    print("Real data experimentation")

elif gtype == "HYP":
    data_path = "./datasets/data_splits/HYP/betweenness/"
    print("Real data experimentation")



#Load training data
print(f"Loading data...")
with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,cc_mat_train = pickle.load(fopen)


with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,cc_mat_test = pickle.load(fopen)


#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_train,list_adj_mod_train = graph_to_adj_clustering(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_mod_test = graph_to_adj_clustering(list_graph_test,list_n_seq_test,list_num_node_test,model_size)



def train(list_adj_train,list_adj_mod_train,list_num_node_train,cc_mat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_mod = list_adj_mod_train[i]
        adj = adj.to(device)
        adj_mod = adj_mod.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_mod)
        true_arr = torch.from_numpy(cc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_mod_test,list_num_node_test,bc_mat_test):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_mod = list_adj_mod_test[j]
        adj=adj.to(device)
        adj_mod = adj_mod.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_mod)
    
        
        true_arr = torch.from_numpy(cc_mat_test[:,j]).float()
        true_val = true_arr.to(device)
    
        kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
        list_kt.append(kt)


    print(f"    Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")
    return np.mean(np.array(list_kt)),np.std(np.array(list_kt))


#Model parameters
hidden = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=None
if gnn_type=="GNN":
    model = GNN_Clustering(ninput=model_size,nhid=hidden,dropout=0.6,num_intermediate_layers=num)
elif gnn_type=="CNN":
    model = CNN_Clustering(ninput=model_size,nhid=hidden,dropout=0.6,num_intermediate_layers=num)
elif gnn_type=="GAT":
    model = GAT_Clustering(ninput=model_size,nhid=hidden,dropout=0.6,num_intermediate_layers=num)
elif gnn_type=="SAGE":
    model = GSAGE_Clustering(ninput=model_size,nhid=hidden,dropout=0.6,num_intermediate_layers=num)
    
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
num_epoch = 15

print("Training")
print(f"Number of epoches: {num_epoch}")
kt_mean=None
std_kt=None
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_mod_train,list_num_node_train,cc_mat_train)

    #to check test loss while training
    with torch.no_grad():
        kt_mean,std_kt=test(list_adj_test,list_adj_mod_test,list_num_node_test,cc_mat_test)
#test on 10 test graphs and print average KT Score and its stanard deviation
#with torch.no_grad():
#    test(list_adj_test,list_adj_mod_test,list_num_node_test,cc_mat_test)
#----------------------------------------------
#Código para guardar resultados
if v!="":
    v=f"_{v}"
if optional!="":
    optional=f"_{optional}"
list_data=list()
list_data.append([kt_mean,std_kt,model.get_num_intermediate_layers(),model.get_gnn_type(),g_hype])
with open(f"results/clustering/{model.get_num_intermediate_layers()}_{model.get_gnn_type()}_{gtype}{optional}{v}_kt.pickle","wb") as fopen2:
        pickle.dump(list_data,fopen2)
print("")
print("Results saved")

    