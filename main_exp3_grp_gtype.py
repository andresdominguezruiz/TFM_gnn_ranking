import subprocess

from utils import call_subprocess

'''
OBJETIVO DE LA EXPERIMENTACIÓN:
- 

'''

#--------------------------------------------------------
#FOR_EXP= válido
#FT= todos menos CLUSTERING (debe ser porque el coef. es 0 siempre)
#HYP no es válido. Es por la forma del grafo, no por la lectura del mismo.
G_TYPE="GRP"
HYPER="s"


#sf_values=[0.01,0.25,0.5,0.75]
#index_values=[1,2,3,4]
index_values=[5,6]
#er_values=[0.00005,0.00125,0.005,0.01]
#grp_values=[100,600,1100,1700]
grp_values=[1,3000]

#IDEA: Estudiar nº de capas óptima en GNN con grafos de entre 100000-50000 nodos(valores del artículo)
for i,j in zip(grp_values,index_values):
    # Construir el comando con el valor de i
    print(f"###########BETWEENNESS PARA {G_TYPE} , CON {HYPER} = {i} ################")
    #RESETEO DE GRAFOS#--------------------------------------
    subprocess.run(f"python central_generate_graph.py --num_graphs 15 --min_nodes 3000 --max_nodes 5000 --g_type {G_TYPE} --{HYPER} {i} --centrality bet", shell=True,cwd="datasets")
    subprocess.run(f"python central_create_dataset.py --split_train 5 --split_test 10 --model_size 5000 --num_copies 20 --g_type {G_TYPE} --centrality bet" ,shell=True,cwd="datasets")

    comando_bet = f"python betweenness.py --g {G_TYPE} --num_intermediate_layer 5 --model_size 5000 --optional_name {HYPER}{j} --g_hype {i}"
    
    # Ejecutar el comando
    call_subprocess(comando_bet)
    print("-------------------------------------------------------------------------------------")
