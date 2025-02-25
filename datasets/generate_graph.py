import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
np.random.seed(1)


#Esto sirve para directamente crear los grafos, pero solo les indican las propiedades
def create_graph(graph_type):

    num_nodes = np.random.randint(5000,10000) #<---AQUI ESTÁ EL MAX NODES DE ESOS GRAFOS

    if graph_type == "ER":
        #Erdos-Renyi random graphs
        p = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes,p = p,directed = True)
        return g_nx

    if graph_type == "SF":
        #Scalefree graphs
        alpha = np.random.randint(40,60)*0.01
        gamma = 0.05
        beta = 1 - alpha - gamma
        g_nx = nx.scale_free_graph(num_nodes,alpha = alpha,beta = beta,gamma = gamma)
        return g_nx


    if graph_type == "GRP":
        #Gaussian-Random Partition Graphs
        s = np.random.randint(200,1000)
        v = np.random.randint(200,1000)
        p_in = np.random.randint(2,25)*0.0001
        p_out = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.gaussian_random_partition_graph(num_nodes,s = s, v = v, p_in = p_in, p_out = p_out, directed = True)
        assert nx.is_directed(g_nx)==True,"Not directed"
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



num_of_graphs = 50
graph_types = ["ER","SF","GRP"]
#-------GENERACIÓN DE DATOS SINTÉTICOS ---------------
'''
Para cada tipo de dato sintético, crea 50 grafos de la siguiente forma:
 1º Cada grafo se crea con nodos y aristas de forma aleatoria, y las variables internas
    de cada tipo indican la probabilidad de que 2 nodos estén unidos por una arista de una forma
    u otra:
       - ER: únicamente depende de la probabilidad de que 2 nodos tengan una arista
       - SF: tiene prob. de que 2 nodos tengan una arista, prob. de añadir un nodo entre la
               arista de 2 nodos, y la prob. de añadir un nuevo nodo conectado a otro.
       - GRP: primero trabajan con el tamaño de los clusters, y luego con la prob.
               de que un nodo se conecte con nodos del mismo cluster (p_in),
               y la prob. de que los nodos de un cluster se conecten con otros (p_out)
 
 2º Le quitan a esos grafos los nodos sin aristas, y si había nodos de ese tipo, actualiza las
    etiquetas de los nodos para asi no tener números sueltos.
 3º Ahora, con esos grafos primero lo pasan a Tipo Graph, para que se le puedan aplicar funciones
    de NetworKit, y luego calculan las centralidades (BET y CLOSE) para cada grafo(lo devuelve)
    en formato Diccionario, CUYO TAMAÑO ES N_i, siendo N_i= nº de nodos del grafo i.
    
    Aplican una función creada por ellos para obtener el coeficiente más exacto posible( es decir
    , que tienen en cuenta muchos decimales al hacerlo de esa forma y no de la forma que 
    NetwortX proporciona)

 4º Al final, guardan en la carpeta /datasets/graphs archivos que tengan como contenido:
        grafo1,diccionario de centralidad,
        grafo2, diccionario de centralidad,
        .
        .
        .
    
    ADVERTENCIA: el código esta hecho para que se ejecute el comando desde /datasets.
'''

for graph_type in graph_types:
    print("###################")
    print(f"Generating graph type : {graph_type}")
    print(f"Number of graphs to be generated:{num_of_graphs}")
    list_bet_data = list()
    list_close_data = list()
    print("Generating graphs and calculating centralities...")
    for i in range(num_of_graphs):
        print(f"Graph index:{i+1}/{num_of_graphs}",end='\r')
        g_nx = create_graph(graph_type) #Aqui se da el 1º paso.
        #----Aquí se da el paso 2º -----------
        if nx.number_of_isolates(g_nx)>0:
            #print("Graph has isolates.")
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)
        #-------------------------------------------
        #-----AQUI OCURRE EL PASO 3º-------------
        g_nkit = nx2nkit(g_nx)
        bet_dict = cal_exact_bet(g_nkit)
        close_dict = cal_exact_close(g_nkit)
        list_bet_data.append([g_nx,bet_dict])
        list_close_data.append([g_nx,close_dict])
        #--------------------------------

    fname_bet = "./graphs/"+graph_type+"_data_bet.pickle"    
    fname_close = "./graphs/"+graph_type+"_data_close.pickle"
    #Aquí ocurre el paso 4º
    with open(fname_bet,"wb") as fopen:
        pickle.dump(list_bet_data,fopen)

    with open(fname_close,"wb") as fopen1:
        pickle.dump(list_close_data,fopen1)
    print("")
    print("Graphs saved")

    

print("End.")


        


