import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
random.seed(10)
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
def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    return g_nkit

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
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:num_train],num_copies = num_copies)

    with open(save_path+"training.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    #For test split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[num_train:num_train+num_test],num_copies = 1)

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