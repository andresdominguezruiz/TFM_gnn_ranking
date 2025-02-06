import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
np.random.seed(1)


#Esto sirve para directamente crear los grafos, pero solo les indican las propiedades
def create_graph(num_nodes):

    #num_nodes debe de ser = 100.000
    #Scalefree graphs
    alpha = np.random.randint(40,60)*0.01
    gamma = 0.05
    beta = 1 - alpha - gamma
    g_nx = nx.scale_free_graph(num_nodes,alpha = alpha,beta = beta,gamma = gamma)
    return g_nx


#Le mete el grafo para añadirle tanto los nodos como las aristas
def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    return g_nkit

def cal_exact_bet(g_nx):

    #exact_bet = nx.betweenness_centrality(g_nx,normalized=True)

    exact_bet = centrality.Betweenness(g_nkit,normalized=True).run().ranking()
    exact_bet_dict = dict()
    for j in exact_bet:
        exact_bet_dict[j[0]] = j[1]
    return exact_bet_dict

def cal_exact_close(g_nx):
    
    #exact_close = nx.closeness_centrality(g_nx, reverse=False)

    exact_close = centrality.Closeness(g_nkit,True,1).run().ranking()

    exact_close_dict = dict()
    for j in exact_close:
        exact_close_dict[j[0]] = j[1]

    return exact_close_dict



num_of_graphs = 5
graph_types = ["SF"]

for graph_type in graph_types:
    print("###################")
    print(f"Generating graph type : {graph_type}")
    print(f"Number of graphs to be generated:{num_of_graphs}")
    list_bet_data = list()
    list_close_data = list()
    print("Generating graphs and calculating centralities...")
    for i in range(num_of_graphs):
        print(f"Graph index:{i+1}/{num_of_graphs}",end='\r')
        g_nx = create_graph(7115) #Cambiar número de nodos para diferentes grafos
        
        if nx.number_of_isolates(g_nx)>0:
            #print("Graph has isolates.")
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)
        g_nkit = nx2nkit(g_nx)
        bet_dict = cal_exact_bet(g_nkit)
        close_dict = cal_exact_close(g_nkit)
        list_bet_data.append([g_nx,bet_dict])
        list_close_data.append([g_nx,close_dict])
    name= "FOR_EXP"
    fname_bet = "./graphs/"+name+"_data_bet.pickle"    
    fname_close = "./graphs/"+name+"_data_close.pickle"

    with open(fname_bet,"wb") as fopen:
        pickle.dump(list_bet_data,fopen)

    with open(fname_close,"wb") as fopen1:
        pickle.dump(list_close_data,fopen1)
    print("")
    print("Graphs saved")

    

print("End.")


        


