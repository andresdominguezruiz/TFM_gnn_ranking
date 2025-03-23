import argparse
import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
import torch
from datasets_tools import *
torch.manual_seed(20)

#creating training/test dataset split for the model

parser = argparse.ArgumentParser(description="Generar dataset.")
parser.add_argument("--split_train", type=int, required=True, help="Número de grafos de entrenamiento a escoger")
parser.add_argument("--split_test", type=int, required=True, help="Número de grafos de test a escoger")
parser.add_argument("--centrality", type=str, required=False, help="Tipo de centralidad",default="all")
parser.add_argument("--model_size", type=int, required=True, help="Tamaño del modelo")
parser.add_argument("--num_copies", type=int, required=True, help="Número de permutaciones")

args = parser.parse_args()

centrality_type=args.centrality

#--------AQUÍ SE DA EL PASO 1º------------------------------
adj_size = args.model_size #MAX_NODES , ESTE REALMENTE NO ES USADO.
graph_types = ["HYP"]
num_train = args.split_train
num_test = args.split_test
#Number of permutations for node sequence
#Can be raised higher to get more training graphs
num_copies = args.num_copies

all_cen=["clustering","bet","close","eigen"]
#Total number of training graphs = 40*6 = 240
#------------------------------------------------------------
def prepare_dataset_creation(g_type,centrality_type,num_train,num_test,num_copies,model_size):
    print("Loading graphs from pickle files...")
    source_file= "./graphs/"+ g_type + "_data_"+centrality_type+".pickle"
    c_type=None
    if centrality_type == "bet":
        c_type="betweenness"
    elif centrality_type == "close":
        c_type="closeness"
    else:
        c_type=centrality_type
    
    save_path="./data_splits/"+g_type+"/"+c_type+"/"
    if g_type=="FOR_EXP":
        get_split_real_data(source_file,"./real_data/Wiki-Vote.txt",num_copies,model_size,save_path,c_type)
    else:
        get_split(source_file,num_train,num_test,num_copies,model_size,save_path)
    print(" Data split saved.")

#----------AQUÍ SE DA EL PASO 2º----------------------------
if centrality_type!="all":
    for g_type in graph_types:
        prepare_dataset_creation(g_type,centrality_type,num_train,num_test,num_copies,adj_size)

else:
    for cen in all_cen:
        for g_type in graph_types:
            prepare_dataset_creation(g_type,cen,num_train,num_test,num_copies,adj_size)
#-------------------------------------------------------------------




