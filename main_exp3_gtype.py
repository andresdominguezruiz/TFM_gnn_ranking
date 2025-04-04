import subprocess

'''
OBJETIVO DE LA EXPERIMENTACIÓN:
- Estudiar resultados de los diferentes tipos de modelo con los diferentes tipos de grafo

RESULTADO: Gráficos sobre las alteraciones de los hiperparámetros a la hora de crear los grafos SF, ER Y GP



'''

#SOLO SE HARÁN PRUEBAS CON SF
#RESETEO DE GRAFOS#--------------------------------------
subprocess.run(f"python central_generate_graph.py --num_graphs 15 --min_nodes 5000 --max_nodes 10000", shell=True,cwd="datasets")
subprocess.run(f"python central_create_dataset.py --split_train 5 --split_test 10 --model_size 10000 --num_copies 20" ,shell=True,cwd="datasets")
#Había que solo incluir SF, el resto TARDA MUCHO
#--------------------------------------------------------
#FOR_EXP= válido
#FT= todos menos CLUSTERING (debe ser porque el coef. es 0 siempre)
#HYP no es válido. Es por la forma del grafo, no por la lectura del mismo.
gtypes=["FOR_EXP"]


sf_values=[0.01,0.25,0.5,0.75]
er_values=[0.00005,0.00125,0.005,0.01]
grp_values=[100,600,1100,1700]

#IDEA: Estudiar nº de capas óptima en GNN con grafos de entre 100000-50000 nodos(valores del artículo)
for i in gtypes:
    # Construir el comando con el valor de i
    print(f"###########BETWEENNESS CON {i} CAPAS INTERMEDIAS################")
    comando_bet = f"python betweenness.py --g {i} --num_intermediate_layer 5 --model_size 10000"
    comando_close=f"python closeness.py --g {i} --num_intermediate_layer 5 --model_size 10000"
    comando_cluster=f"python local_clustering.py --g {i} --num_intermediate_layer 5 --model_size 10000"
    comando_page=f"python page_rank.py --g {i} --num_intermediate_layer 5 --model_size 10000"
    
    # Ejecutar el comando
    #subprocess.run(comando_bet, shell=True)
    print(f"###########CLOSENESS CON {i} CAPAS INTERMEDIAS################")
    #subprocess.run(comando_close, shell=True)
    print(f"########### LOCAL CLUSTERING CON {i} CAPAS INTERMEDIAS################")
    subprocess.run(comando_cluster, shell=True)
    print(f"###########AUTOVALOR CON {i} CAPAS INTERMEDIAS################")
    subprocess.run(comando_page, shell=True)
    print("-------------------------------------------------------------------------------------")
