from networkit import *
import networkx as nx
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import pickle
import scipy.sparse as sp
import copy
import random
import numpy as np
import torch


def get_out_edges(g_nkit,node_sequence):
    global all_out_dict
    all_out_dict = dict()
    for all_n in node_sequence:
        all_out_dict[all_n]=set()
        
    for all_n in node_sequence:
            _ = g_nkit.forEdgesOf(all_n,nkit_outedges)
            
    return all_out_dict

def get_in_edges(g_nkit,node_sequence):
    global all_in_dict
    all_in_dict = dict()
    for all_n in node_sequence:
        all_in_dict[all_n]=set()
        
    for all_n in node_sequence:
            _ = g_nkit.forInEdgesOf(all_n,nkit_inedges)
            
    return all_in_dict


def nkit_inedges(u,v,weight,edgeid):
    all_in_dict[u].add(v)


def nkit_outedges(u,v,weight,edgeid):
    all_out_dict[u].add(v)

    

def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    assert g_nx.number_of_nodes()==g_nkit.numberOfNodes(),"Number of nodes not matching"
    assert g_nx.number_of_edges()==g_nkit.numberOfEdges(),"Number of edges not matching"
        
    return g_nkit

def clique_check_for_eigen(index,node_sequence,all_out_dict,all_in_dict):
    '''
    El objetivo de esta función es decir si un nodo_i es importante (True) o no (False).
    No será importante el nodo SI sólo presenta aristas de salida.
    '''
    node = node_sequence[index]
    in_nodes = all_in_dict[node] # nodos que entran en nodo_i
    if len(in_nodes)<=0:
        return False

    return True

def clique_check_for_clustering(index,node_sequence,all_out_dict,all_in_dict):
    '''
    El objetivo de esta función es decir si un nodo_i es importante (True) o no (False).
    No será importante el nodo SI sólo presenta aristas de salida.
    '''
    node = node_sequence[index]
    in_nodes = all_in_dict[node] # nodos que entran en nodo_i
    out_nodes = all_out_dict[node] # nodos que salen de nodo_i
    
    if (len(in_nodes) + len(out_nodes))<=1:
        return True

    return False




def clique_check(index,node_sequence,all_out_dict,all_in_dict):
    '''
    El objetivo de esta función es decir si un nodo_i es ZP o no, viendo si se cumple una de las
    2 condiciones:
    
        - CONDICIÓN 1: Que el nodo_i sólo tenga aristas de entrada o de salida
        - CONDICIÓN 2: Que todos los nodos de entrada del nodo_i compartan nodos de salida con el nodo_i
    '''
    node = node_sequence[index]
    in_nodes = all_in_dict[node] # nodos que entran en nodo_i
    out_nodes = all_out_dict[node] # nodos que salen del nodo_i
    #SOBRENTIENDE QUE el nodo_i es un ZP.
    #Si algún nodo de entrada del nodo_i que NO comparta nodo de salida con nodo_i, entonces
    #NO es ZP.
    #TIENE TANTO EN CUENTA LA CONDICIÓN 1 COMO LA 2, PORQUE CUANDO SÓLO TIENE NODOS DE ENTRADA,
    #EL issubset de un conjunto vacio es True.
    for in_n in in_nodes:
        tmp_out_nodes = set(out_nodes)
        tmp_out_nodes.discard(in_n)
        if tmp_out_nodes.issubset(all_out_dict[in_n]) == False:
            return False
    
    #SI todos los nodos de entrada del nodo_i comparten nodo de salida con el nodo_i, entonces nodo_i es un ZeroPath
    return True

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


#---------------OBTENCIÓN DE MATRICES DE ADJACENCIA------------------




def graph_to_adj_bet(list_graph,list_n_sequence,list_node_num,model_size):
    '''
1º Se va recorriendo la lista de grafos, y lo que se hace es crear un grafo nuevo a partir
    de las aristas y nodos del grafo leido (ESTO SE HACE PARA PONERLO EN OTRO FORMATO GRAFO MÁS
    MANEJABLE)
    Una vez hecho eso, primero se le eliminan los bucles, y se crea la primera versión de dos variables:
        - adj_temp: será la matriz de adjacencia DEL ELEMENTO i, que tiene el tamaño N_i x N_i, Y SIGUIENDO LA ORDENACIÓN
                    de los nodos permutados (el list_n_sequence de ese grafo)
        - adj_temp_t: será la matriz de adjacencia transpuesta DEL ELEMENTO i.
    
2º RESUMEN: A partir de A y At ==> A modificado y At modificado (TODAVÍA CON TAMAÑO N_i x N_i), quitandole los Node Zero Path.
RECUERDA, se les quita estos nodos para sólo transmitir la información de los nodos que participan
en los camínos mínimos.

Lo que se hace para obtener las matrices modificadas es lo siguiente:
    - 1. Se obtienen las listas de grados de A y A_t, y multiplicando sus valores se obtiene
        la matriz de grados. Ahora, a esa matriz la actualizas para tener una matriz de 1s y 0s
        A esa matriz, se le obtienen aquellos nodos que tengan grado >0, y lo guarda en una lista.
    - 2. Ahora, obtienen el diccionario de nodos de salida y de entrada de los nodos del elemento_i,
        y con esos diccionarios va leyendo todos los nodos con grado>0 y comprueba si pertenece a ZP
        o no.
            Si pertenece, en la matriz de grados pone ese nodo a 0.
            Si no, siguiente.
    - 3. Al final, en degree_arr voy a tener una matriz formada por 1s y 0s, pero con 0s en los Nzp.
        Se multiplica esta matriz con las matrices de adjacencia normal y transpuesta y ya obtienes las matrices
        modificadas.
            OJO, siguen teniendo tamaño (N_i x N_i)

3º Después, lo que se hace es una EXPANSIÓN DE LAS MATRICES DE ADYACENCIA PARA QUE TENGAN
    COMO TAMAÑO N_MAX x N_MAX. 
    La expansión se hace 1º creando una matriz N_MAX x N_MAX llena de 0s, y desde la posición (0,0)
    se va rellenando con la matriz de adyacencia modificada.

4º El último paso es guardar las matrices expandidas de adyacencia normal y transpuesta en la lista,
y luego se sigue con el siguiente grafo y su secuencia de nodos.
    '''

    list_adjacency = list()
    list_adjacency_t = list()
    list_degree = list()
    max_nodes = model_size
    zero_list = list()
    list_rand_pos = list()
    list_sparse_diag = list()
    #-----------------AQUI OCURRE EL PASO 1º-------------------------------------
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r')
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        #self_loops = [i for i in graph.selfloop_edges()]
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i] #La secuencia de nodos del grafo(contiene los nodos permutados)

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        #--------------------------------------------------------------
        
        #--------------AQUÍ OCURRE PASO 2º---------------------
        arr_temp1 = np.sum(adj_temp,axis=1) #Esto me devuelve el array de los grados
        #Ej: [[0,1,1],[1,0,0],[1,0,0]] => [2,1,1]
        arr_temp2 = np.sum(adj_temp_t,axis=1)
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)
        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq)


        #Aqui te busca los Nzp y te va quitando los que no necesitas
        for index in non_zero_ind:
           
            is_zero = clique_check(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == True:
              
                degree_arr[index,0]=0.0
        #Al final de este proceso, degree_arr contendrá una matriz DIAGONAL de 1s y 0s, y con 0s en aquellos 
        #que sean Zero Path.        
        adj_temp = adj_temp.multiply(csr_matrix(degree_arr)) #AQUI YA OBTIENES Amod y Atmod
        adj_temp_t = adj_temp_t.multiply(csr_matrix(degree_arr))
        
        #--------------------------------------------------------
                

        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor

        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_t = csr_matrix(adj_temp_t)
        adj_mat_t = sp.block_diag((top_mat,adj_temp_t,bottom_mat))
        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_t = sparse_mx_to_torch_sparse_tensor(adj_mat_t)
        list_adjacency_t.append(adj_mat_t)
    print("")          
    return list_adjacency,list_adjacency_t

def graph_to_adj_close(list_graph,list_n_sequence,list_node_num,model_size,print_time=False):
    '''
    LA DIFERENCIA CON EL MÉTODO DE CLOSENESS ES QUE EN ESE TE DEVUELVEN:
    - la lista de matrices de adyacencia REESCALADAS A TAMAÑO N_MAX x N_MAX, pero sin quitar los Nzp.
    - la lista de matrices de adyacencia modificada con tamaño N_MAX x N_MAX.
    '''

    list_adjacency = list()
    list_adjacency_mod = list()
    list_degree = list()
    max_nodes = model_size
    zero_list = list()
    list_rand_pos = list()
    list_sparse_diag = list()
    
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r')
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]
        
        

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        arr_temp1 = np.sum(adj_temp,axis=1)
        arr_temp2 = np.sum(adj_temp_t,axis=1)
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)

        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq)

        
        for index in non_zero_ind:
           
            is_zero = clique_check(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == True:
              
                degree_arr[index,0]=0.0


        #modify the in-degree matrix for different layers

        degree_arr = degree_arr.reshape(1,node_num)
 

        #for out_degree
        adj_temp_mod = adj_temp.multiply(csr_matrix(degree_arr))


        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor
        
        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_mod = csr_matrix(adj_temp_mod)
        adj_mat_mod = sp.block_diag((top_mat,adj_temp_mod,bottom_mat))

        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_mod = sparse_mx_to_torch_sparse_tensor(adj_mat_mod)
        list_adjacency_mod.append(adj_mat_mod)

    print("")        
    return list_adjacency,list_adjacency_mod

def graph_to_one_hot_matrix(list_graph,list_n_sequence,list_node_num,model_size,print_time=False):
    '''
    Aunque no se haya probado, este tipo de entrada NO tendría sentido, ya que
    no proporciona información de nodos vecinos ni nada.
    '''

    list_adjacency = list()
    list_adjacency_mod = list()
    max_nodes = model_size
    list_rand_pos = list()
    
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r')
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]
        node_num = list_node_num[i]
        
        # Determinar el número de bits necesarios
        n_bits = (model_size - 1).bit_length()  # Equivalente a ceil(log2(n_nodos))

        # Inicializar matriz binaria
        binary_matrix_list = []

# Llenar la lista con la representación binaria de los índices de los nodos
        for nodo in node_sequence:
            bin_repr = list(map(int, format(nodo, f'0{n_bits}b')))
            binary_matrix_list.append(bin_repr)
        
        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))

# Convertir la lista en una matriz NumPy
        binary_matrix_np = np.array(binary_matrix_list)

# Convertir la matriz NumPy a una matriz dispersa CSR de Scipy
        binary_matrix_sparse = csr_matrix(binary_matrix_np)
        
        mat = sp.block_diag((top_mat,binary_matrix_sparse,bottom_mat))

        list_adjacency.append(mat)
        list_adjacency_mod.append(mat)

    print("")        
    return list_adjacency,list_adjacency_mod



def graph_to_adj_clustering(list_graph,list_n_sequence,list_node_num,model_size):
    '''
    Output: Lista de matrices de Ady. modificada y sin modificar.(como el de closeness)
    Aquí en vez de trabajar con Nzp, se trabajará con otro N.
    Si un nodo sólo tiene 1 arista, no importa que sea de salida o de entrada, implica que su clustering es 0
    '''

    list_adjacency = list()
    list_adjacency_mod = list()
    max_nodes = model_size
    list_rand_pos = list()
    
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r')
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        arr_temp1 = np.sum(adj_temp,axis=1)
        arr_temp2 = np.sum(adj_temp_t,axis=1)
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)

        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq)

        #AQUI SE IGNORARÁN OTROS NODOS---------
        for index in non_zero_ind:
           
            is_zero = clique_check_for_clustering(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == True: #Si ese nodo SOLO tiene una arista, poner a 0
                degree_arr[index,0]=0.0


        #modify the in-degree matrix for different layers

        degree_arr = degree_arr.reshape(1,node_num)
 

        #for out_degree
        adj_temp_mod = adj_temp.multiply(csr_matrix(degree_arr))


        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor
        
        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_mod = csr_matrix(adj_temp_mod)
        adj_mat_mod = sp.block_diag((top_mat,adj_temp_mod,bottom_mat))

        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_mod = sparse_mx_to_torch_sparse_tensor(adj_mat_mod)
        list_adjacency_mod.append(adj_mat_mod)

    print("")        
    return list_adjacency,list_adjacency_mod

#-----------------OBTENCIÓN DE MATRICES DE ADY. PARA AUTOVALOR Y CLUSTERING---------
def graph_to_adj_eigen(list_graph,list_n_sequence,list_node_num,model_size):
    '''
    Output: Lista de matrices de Ady. modificada y sin modificar.(como el de closeness)
    Aquí en vez de trabajar con Nzp, se trabajará con otro N.
    Si un nodo sólo tienen aristas de salida, el autovalor de ese nodo tenderá a 0, por lo que no
    aportan mucha información relevante. Por lo que se ignorarán esos nodos.
    '''

    list_adjacency = list()
    list_adjacency_mod = list()
    max_nodes = model_size
    list_rand_pos = list()
    
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r')
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        arr_temp1 = np.sum(adj_temp,axis=1)
        arr_temp2 = np.sum(adj_temp_t,axis=1)
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)

        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq)

        #AQUI SE IGNORARÁN OTROS NODOS---------
        for index in non_zero_ind:
           
            is_zero = clique_check_for_eigen(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == False: #Si ese nodo NO es importante al tener sólo nodos de salida,
                #entonces ignoramelo en la matriz de adyacencia.
                degree_arr[index,0]=0.0


        #modify the in-degree matrix for different layers

        degree_arr = degree_arr.reshape(1,node_num)
 

        #for out_degree
        adj_temp_mod = adj_temp.multiply(csr_matrix(degree_arr))


        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor
        
        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_mod = csr_matrix(adj_temp_mod)
        adj_mat_mod = sp.block_diag((top_mat,adj_temp_mod,bottom_mat))

        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_mod = sparse_mx_to_torch_sparse_tensor(adj_mat_mod)
        list_adjacency_mod.append(adj_mat_mod)

    print("")        
    return list_adjacency,list_adjacency_mod





def ranking_correlation(y_out,true_val,node_num,model_size):
    print(true_val)
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()


    kt,_ = kendalltau(predict_arr[:node_num],true_arr[:node_num])

    return kt


def loss_cal(y_out,true_val,num_nodes,device,model_size):

    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))
    
    _,order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes*20 #Nº de pares de nodos a tener en cuenta.

    ind_1 = torch.randint(0,num_nodes,(sample_num,)).long().to(device) #aquí me escoge 20*N nodos
    ind_2 = torch.randint(0,num_nodes,(sample_num,)).long().to(device) # y aquí otros 20*N nodos
    

    rank_measure=torch.sign(-1*(ind_1-ind_2)).float()
        
    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)
        

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1,input_arr2,rank_measure)
 
    return loss_rank

