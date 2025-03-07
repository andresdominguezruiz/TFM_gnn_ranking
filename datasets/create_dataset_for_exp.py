import argparse
import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
random.seed(10)
from datasets_tools import *


# Configuración de argumentos desde la terminal
parser = argparse.ArgumentParser(description="Generar grafos y calcular centralidades.")
parser.add_argument("--model_size", type=int, required=True, help="Tamaño del modelo")
parser.add_argument("--num_copies", type=int, required=True, help="Número de copias para las permutaciones del entrenamiento")
parser.add_argument("--file", type=str, required=True, help="Archivo .txt del grafo de prueba")

args = parser.parse_args()
model_size = args.model_size
num_copies = args.num_copies
test_file=args.file

print("Loading graphs from pickle files...")
bet_source_file = "./graphs/"+ "FOR_EXP" + "_data_bet.pickle"
close_source_file = "./graphs/"+ "FOR_EXP" + "_data_close.pickle"

    #paths for saving splits
save_path_bet = "./data_splits/"+"FOR_EXP"+"/betweenness/"
save_path_close = "./data_splits/"+"FOR_EXP"+"/closeness/"
#SPLIT DE BETWEENNESS:
get_split_real_data(bet_source_file,test_file,num_copies,model_size,save_path_bet,"betweenness")

#SPLIT DE CLOSENESS:
get_split_real_data(close_source_file,test_file,num_copies,model_size,save_path_close,"closeness")

print("Finish")

