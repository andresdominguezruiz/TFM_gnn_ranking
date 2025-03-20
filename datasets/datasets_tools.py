import networkx as nx
import pickle
import numpy as np
import time
import glob
import random

from networkit import *

#-----------CREACIÓN DE DATASETS -------------------------
'''
1º Para crear los datasets, primero tienen en cuenta:
    - El tamaño del modelo (adj_size), para actualizar las matrices de centralidad con ese tamaño
    - La separación de los elementos para entrenamiento y de tests. Por ejemplo: si han generado
        50 grafos desde generate_graph, de esos 50 hacen la separación, 40 para entrenamiento
         y 10 para tests(VER SI LES SIRVE A ELLOS TAMBIÉN COMO VALIDACIÓN).
    - Tras decir la separación, se define EL Nº DE PERMUTACIONES A REALIZAR (num_copies)

2º Para cada tipo de grafo sintético, preparan la lectura del archivo que contiene 
    grafos, diccionario de centralidad, y la escritura se realizará en:
        /datasets/data_split/TIPO_GRAFO/CENTRALIDAD/, en donde guardarán el dataset de entrenamiento
        y el dataset de test.

3º En la función **get_split** lo que hacen es separar la creación del dataset de entrenamiento
    de la creación del dataset de test. El dataset de test NO PRESENTARÁ PERMUTACIONES. Pero al hacer la escritura,
    el resultado será que TODO dato tendrá tamaño N_MAX, siendo N_MAX el tamaño del modelo. Los datos
    que contendrán los datasets serán:
        lista de grafos / lista de la secuencia de nodos / lista del nº de nodos de cada grafo 
        / matriz de centralidad.
        
        Cada elemento tendrá por lo tanto: grafo-i, sencuencia-i, nº de nodos del grafo-i,
        lista de centralidad-i. La lista/array está preparada para que añada 0s para aumentar la dimesión
        a N_MAX.

4º Ahora, en la función **create_dataset**, lo primero que se hace es inicializar los elementos
que se van a guardar:
    - list_graphs: lo ponen como lista vacía, y será donde se guarden los Tipo Graph de los elementos
    (recuerda que cada elemento sería un grafo)
    - list_node_num: lo ponen como lista vacía, y será donde guarden el tamaño del elemento-i
    - list_n_sequence: lo ponen como lista vacía, y será donde guarden el nº de la secuencia del elemento.
                        Esto es el nº de la permutación al que pertenece el elemento. Sirve para tener un
                        orden de los elementos del dataset.
    - cent_mat: lo inicializan con 0s , y es una matriz tamaño_modelo x n º elementos totales(nº grafos * nº de copias)

5º Después de la inicialización, se va recorriendo la lista de grafos que se separaron, y dentro
 de ese bucle lo que se hace es:
    - Hacer num_copies permutaciones de los nodos de ese grafo
    - En cada permutación, se obtiene un orden de nodos diferente, lo cuál se aprovecha
        para añadir en la matriz cent_mat [indice del nodo, nº de permutación]= centralidad del nodo
    - A la vez, se va guardando en list_graphs, list_node_num y list_n_sequency sus elementos.
     El grafo realmente no se toca, lo que se toca es su lista de nodos, la cuál se guarda su permutación
    en list_n_sequency.

6º Una vez recorrido todos los grafos num_copies veces, se hace una REORDENACIÓN DE TODAS LAS LISTAS.
    Es decir, de todos los elementos (total_num), se crea una ordenación a seguir para las listas.
    EJ: si las listas estaban de esta forma=> elemento1_1,elemento1_2,...elemento1_n, elemento2_1....
    , pasan a la siguiente forma: elemento35_2, elemento20_5,.....
    
    Es decir, RANDOMIZA EL DATASET.
    
    

'''
def nx2nkit(g_nx,is_directed=True):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=is_directed)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    return g_nkit

def obtain_dictionary(elements):
    dictionary=dict()
    for j in elements:
        dictionary[j[0]] = j[1]
    return dictionary

def cal_exact_local_transitivity(g_nkit):
    print(type(g_nkit))
    exact_local_trans=centrality.LocalClusteringCoefficient(g_nkit).run().ranking()
    exact_dict=obtain_dictionary(exact_local_trans)
    return exact_dict

def cal_exact_page_rank(g_nkit):
    exact_page_rank= centrality.PageRank(g_nkit,normalized=True).run().ranking()
    exact_page_rank_dict = obtain_dictionary(exact_page_rank)
    return exact_page_rank_dict

def cal_exact_bet(g_nkit):

    #exact_bet = nx.betweenness_centrality(g_nx,normalized=True)

    exact_bet = centrality.Betweenness(g_nkit,normalized=True).run().ranking()
    exact_bet_dict = dict()
    for j in exact_bet:
        exact_bet_dict[j[0]] = j[1]
    return exact_bet_dict

def cal_exact_close(g_nkit):
    
    #exact_close = nx.closeness_centrality(g_nx, reverse=False)

    exact_close = centrality.Closeness(g_nkit,True,1).run().ranking()

    exact_close_dict = dict()
    for j in exact_close:
        exact_close_dict[j[0]] = j[1]

    return exact_close_dict


def load_txt_graph(file_path, centrality_type):
    G = nx.DiGraph()

    with open(file_path, "r") as f:
        edges = set()
        nodes = set()
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            node1, node2 = map(int, line.split())
            edges.add((node1, node2))
            nodes.update([node1, node2])

    G.add_edges_from(edges)
    G.add_nodes_from(nodes)

    mapping = {old: new for new, old in enumerate(G.nodes())}
    G_reset = nx.relabel_nodes(G, mapping)
    #-----Hay que pasarlo al otro formato de Graph para el cálculo exacto de las centralidades
    g_nkit=nx2nkit(G_reset)
    if centrality_type == "betweenness":
        centrality = cal_exact_bet(g_nkit)
    elif centrality_type == "closeness":
        centrality = cal_exact_close(g_nkit)
    else:
        raise ValueError("centrality_type debe ser 'betweenness' o 'closeness'")
    
    return G_reset, centrality



def reorder_list(input_list,serial_list):
    new_list_tmp = [input_list[j] for j in serial_list]
    return new_list_tmp

def create_dataset(list_data,num_copies,model_size):
    #-----AQUI OCURRE EL PASO 4º---------------------------------------
    adj_size = model_size #MAX_NODES
    num_data = len(list_data)
    total_num = num_data*num_copies
    cent_mat = np.zeros((adj_size,total_num),dtype=np.float) #AQUI PREPARAN YA LA MATRIZ CON 0s
    #AHORA, cent_mat= matriz de tamaño adj_size x total_num(nºdatos*permutaciones),
    #indicando que para cada fila(que implica un elemento) hay adj_size columnas que indicarán
    #las centralidades de cada nodo de ese elemento, y en casos en los que un elemento(un grafo)
    #no tenga tantos nodos, se mantendrán los 0s de esos índices.
    list_graph = list()
    list_node_num = list()
    list_n_sequence = list()
    mat_index = 0
    #------------------------------------------------------
    #-------------AQUI OCURRE PASO 5º----------------------------
    for g_data in list_data: #Va recorriendo por cada grafo
        #Para cada grafo, hace las num_copies permutaciones de sus nodos, y va moviendo sus centralidades
        #a esos nodos, además de guardar el grafo original, el tamaño, los nodos permutados.

        graph, cent_dict = g_data
        nodelist = [i for i in graph.nodes()] #nodeList de tamaño N_i
        assert len(nodelist)==len(cent_dict),"Number of nodes are not equal"
        node_num = len(nodelist)

        for i in range(num_copies):
            tmp_nodelist = list(nodelist)
            random.shuffle(tmp_nodelist) #tmp_nodeList= lista de nodos desordenados
            list_graph.append(graph)
            list_node_num.append(node_num)
            list_n_sequence.append(tmp_nodelist)

            for ind,node in enumerate(tmp_nodelist):
                #Con esto, se lee del diccionario del elemento la centralidad del nodo, y se guarda
                # en la matriz.
                #ESTO ESTA HECHO DE TAL FORMA QUE PARA CADA ELEMENTO DE LA LISTA, SE VAYAN GUARDANDO
                #SUS CENTRALIDADES PERMUTADAS num_copies VECES.
                cent_mat[ind,mat_index] = cent_dict[node]
            mat_index +=  1

    #----------------------------------------
    #----------AQUÍ OCURRE EL PASO 6º---------------------
    serial_list = [i for i in range(total_num)]
    random.shuffle(serial_list)
    list_graph = reorder_list(list_graph,serial_list)
    list_n_sequence = reorder_list(list_n_sequence,serial_list)
    list_node_num = reorder_list(list_node_num,serial_list)
    cent_mat_tmp = cent_mat[:,np.array(serial_list)]
    cent_mat = cent_mat_tmp
    #---------------------------------------------------------------------
    return list_graph, list_n_sequence, list_node_num, cent_mat

#AQUÍ OCURRE EL PASO 3º----------------------
def get_split(source_file,num_train,num_test,num_copies,adj_size,save_path):

    with open(source_file,"rb") as fopen:
        list_data = pickle.load(fopen)

    num_graph = len(list_data)
    assert num_train+num_test == num_graph,"Required split size doesn't match number of graphs in pickle file."
    
    #For training split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:num_train],num_copies,adj_size)

    with open(save_path+"training.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    #For test split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[num_train:num_train+num_test],1,adj_size)

    with open(save_path+"test.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)


def get_split_real_data(source_train_file,source_real_file,num_copies,model_size,save_path,centrality):

    with open(source_train_file,"rb") as fopen:
        list_data = pickle.load(fopen)
    
    #For training split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data,num_copies,model_size)

    with open(save_path+"training.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    #For test split
    list_test_data=list()
    real_graph,centrality_dicc=load_txt_graph(source_real_file,centrality)
    list_test_data.append([real_graph,centrality_dicc])
    
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_test_data,1,model_size)

    with open(save_path+"test.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)
        

def create_graph(graph_type,min_nodes=5000,max_nodes=10000,is_directed=True):
    '''
    ER y GRP==> tipo DiGraph (2 pares de nodos sólo pueden estar conectados por 1 arista, no más)
    
    SF==> tipo MultiDiGraph (2 pares de nodos pueden tener muchas aristas a la vez entre si)
    '''

    num_nodes = np.random.randint(min_nodes,max_nodes) #<---AQUI ESTÁ EL MAX NODES DE ESOS GRAFOS
    #https://networkx.org/documentation/stable/reference/generators.html
     #--------ADICIONALES----------------------------------------
    if graph_type == "FT": #Full rary Tree
        r= np.random.randint(10,45)
        g_nx=nx.generators.full_rary_tree(r,num_nodes)
        return g_nx
    
    
    if graph_type == "TU": #Turan Graph
        r= np.random.randint(5,num_nodes-1)
        g_nx=nx.generators.turan_graph(num_nodes,r)
        return g_nx
    
    #------------------------------------------------------------

    if graph_type == "ER":
        #Erdos-Renyi random graphs
        p = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes,p = p,directed = is_directed)
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
        g_nx = nx.generators.gaussian_random_partition_graph(num_nodes,s = s, v = v, p_in = p_in, p_out = p_out, directed = is_directed)
        assert nx.is_directed(g_nx)==True,"Not directed"
        return g_nx


def complete_generation(graph_types,num_of_graphs,min_nodes=5000,max_nodes=10000):
    for graph_type in graph_types:
        print("###################")
        print(f"Generating graph type : {graph_type}")
        print(f"Number of graphs to be generated:{num_of_graphs}")
        list_bet_data = list()
        list_close_data = list()
        list_eigen_data=list()
        list_clustering_data=list()
        print("Generating graphs and calculating centralities...")
        for i in range(num_of_graphs):
            print(f"Graph index:{i+1}/{num_of_graphs}",end='\r')
            g_nx = create_graph(graph_type,min_nodes,max_nodes) #Aqui se da el 1º paso.
            #----Aquí se da el paso 2º -----------
            if nx.number_of_isolates(g_nx)>0:
                g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
                g_nx = nx.convert_node_labels_to_integers(g_nx) 
        #-------------------------------------------
        #-----AQUI OCURRE EL PASO 3º-------------
            g_nkit = nx2nkit(g_nx)
            g_nkit_for_clus= nx2nkit(g_nx,False)
            bet_dict = cal_exact_bet(g_nkit)
            close_dict = cal_exact_close(g_nkit)
            eigen_dict=cal_exact_page_rank(g_nkit)
            Graph.removeSelfLoops(g_nkit_for_clus)
            clus_dict=cal_exact_local_transitivity(g_nkit_for_clus)
            list_bet_data.append([g_nx,bet_dict])
            list_close_data.append([g_nx,close_dict])
            list_eigen_data.append([g_nx,eigen_dict])
            list_clustering_data.append([g_nx,clus_dict])
        
        #--------------------------------

        fname_bet = "./graphs/"+graph_type+"_data_bet.pickle"    
        fname_close = "./graphs/"+graph_type+"_data_close.pickle"
        fname_eigen = "./graphs/"+graph_type+"_data_eigen.pickle"
        fname_clus= "./graphs/"+graph_type+"_data_clustering.pickle"
        #Aquí ocurre el paso 4º
        with open(fname_bet,"wb") as fopen:
            pickle.dump(list_bet_data,fopen)

        with open(fname_close,"wb") as fopen1:
            pickle.dump(list_close_data,fopen1)
    
        with open(fname_eigen,"wb") as fopen2:
            pickle.dump(list_eigen_data,fopen2)
        
        with open(fname_clus,"wb") as fopen2:
            pickle.dump(list_clustering_data,fopen2)
        
        print("")
        print("Graphs saved")

def generation_per_centrality(graph_types,num_of_graphs,centrality,min_nodes=5000,max_nodes=10000):
    for graph_type in graph_types:
        print(f"########## CENTRALITY TYPE: {centrality} #########")
        print(f"Generating graph type : {graph_type}")
        print(f"Number of graphs to be generated:{num_of_graphs}")
        list_data=list()
        print("Generating graphs and calculating centralities...")
        for i in range(num_of_graphs):
            print(f"Graph index:{i+1}/{num_of_graphs}",end='\r')
            g_nx = create_graph(graph_type,min_nodes,max_nodes) #Aqui se da el 1º paso.
            #----Aquí se da el paso 2º -----------
            if nx.number_of_isolates(g_nx)>0:
                g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
                g_nx = nx.convert_node_labels_to_integers(g_nx)
        #-------------------------------------------
            g_nkit = None
            if centrality == "clustering":
                #el cálculo del clustering local SÓLO FUNCIONA CON GRAFOS NO DIRIGIDOS
                print("Diferente")
                g_nkit = nx2nkit(g_nx,False)
            else:
                g_nkit=nx2nkit(g_nx)
            dictionary=None
            if centrality == "bet":
                dictionary=cal_exact_bet(g_nkit)
            elif centrality == "close":
                dictionary=cal_exact_close(g_nkit)
            elif centrality == "eigen":
                dictionary=cal_exact_page_rank(g_nkit)
            elif centrality == "clustering":
                #Con SF da error, hay que investigar el motivo
                Graph.removeSelfLoops(g_nkit)
                dictionary=cal_exact_local_transitivity(g_nkit)
            else:
                raise ValueError(f"La centralidad {centrality} no es usada en el sistema")
            
            list_data.append([g_nx,dictionary])
        #--------------------------------

        fname = "./graphs/"+graph_type+"_data_"+centrality+".pickle"    
        #Aquí ocurre el paso 4º
        with open(fname,"wb") as fopen:
            pickle.dump(list_data,fopen)
        print("")
        print("Graphs saved")