import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
from datasets_tools import *
import torch
import argparse
from datasets_tools import *



parser = argparse.ArgumentParser(description="Generar grafos y calcular centralidades.")
parser.add_argument("--num_graphs", type=int, required=True, help="Número de grafos a generar")
parser.add_argument("--centrality", type=str, required=False, help="Tipo de centralidad",default="all")
parser.add_argument("--min_nodes", type=int, required=False, help="Número mínimo de nodos",default=5000)
parser.add_argument("--max_nodes", type=int, required=False, help="Número máximo de nodos",default=10000)
parser.add_argument("--alpha", type=float, required=False, help="Alpha a la hora de crear grafos SF",default=None)
parser.add_argument("--p", type=float, required=False, help="P a la hora de crear grafos ER",default=None)
parser.add_argument("--s", type=float, required=False, help="S a la hora de crear grafos GRP",default=None)


args = parser.parse_args()
num_of_graphs = args.num_graphs
centrality_type=args.centrality
mini=args.min_nodes
maxi=args.max_nodes

alpha=args.alpha
p=args.p
s=args.s


#EL SF DEVUELVE UN MULTIDRIGRAPH, Y ESE DA PROBLEMAS CON EL CLUSTERING
graph_types = ["SF"]
#graph_types = ["SF"]
#centrality_types = ["bet","close","eigen","clustering"]

if centrality_type == "all":
    complete_generation(graph_types,num_of_graphs,mini,maxi,alpha,p,s)
else:
    generation_per_centrality(graph_types,num_of_graphs,centrality_type,mini,maxi)

print("End.")


        


