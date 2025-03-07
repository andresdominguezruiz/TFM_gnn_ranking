import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
from datasets_tools import *
np.random.seed(1)
import argparse
from datasets_tools import *



parser = argparse.ArgumentParser(description="Generar grafos y calcular centralidades.")
parser.add_argument("--num_graphs", type=int, required=True, help="Número de grafos a generar")
parser.add_argument("--centrality", type=str, required=False, help="Tipo de centralidad",default="all")

args = parser.parse_args()
num_of_graphs = args.num_graphs
centrality_type=args.centrality


#EL SF DEVUELVE UN MULTIDRIGRAPH, Y ESE DA PROBLEMAS CON EL CLUSTERING
graph_types = ["SF","ER","GRP"]
#centrality_types = ["bet","close","eigen","clustering"]

if centrality_type == "all":
    complete_generation(graph_types,num_of_graphs)
else:
    generation_per_centrality(graph_types,num_of_graphs,centrality_type)

print("End.")


        


