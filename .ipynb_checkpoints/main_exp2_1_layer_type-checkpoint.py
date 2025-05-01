'''
OBJ: Estudiar comportamiento de la evolución de KT POR ÉPOCA de un tipo de modelo y grafo

RESULTADO: Mostrar gráfica de varios modelos de bet con diferente cantidad de capas y
estudiando el KT de cada época

'''

import subprocess

from utils import call_subprocess


#SOLO SE HARÁN PRUEBAS CON SF
#RESETEO DE GRAFOS#--------------------------------------
subprocess.run(f"python central_generate_graph.py --num_graphs 15 --min_nodes 3000 --max_nodes 5000", shell=True,cwd="datasets")
subprocess.run(f"python central_create_dataset.py --split_train 5 --split_test 10 --model_size 5000 --num_copies 100" ,shell=True,cwd="datasets")
#Había que solo incluir SF, el resto TARDA MUCHO
#--------------------------------------------------------
num_capas=[10]
#IDEA: Estudiar nº de capas óptima en GNN con grafos de entre 100000-50000 nodos(valores del artículo)
for num in num_capas:
    # Construir el comando con el valor de i
    print(f"########### CLUSTERING CON {num} CAPAS INTERMEDIAS PARA CNN ################")
    comando = f"python local_clustering.py --g SF --num_intermediate_layer {num} --model_size 5000 --gnn CNN "
    # Ejecutar el comando
    call_subprocess(comando)

    print(f"########### CLUSTERING CON {num} CAPAS INTERMEDIAS PARA GAT ################")
    comando = f"python local_clustering.py --g SF --num_intermediate_layer {num} --model_size 5000 --gnn GAT "
    # Ejecutar el comando
    call_subprocess(comando)
    print("-------------------------------------------------------------------------------------")
