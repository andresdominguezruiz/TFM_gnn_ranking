import subprocess

'''
OBJETIVO DE LA EXPERIMENTACIÓN:
- Estudiar resultados de los diferentes tipos de modelo con los diferentes tipos de grafo

RESULTADO: Una tabla Tipo grafo|Tipo modelo en la que se muestran los Kts+std obtenidos.

Evolución vista= Con Turan Graph aumenta tanto la complejidad de que ni en varias horas ha generado 1 grafo pequeño.
Realmente, los demás tipos de grafos son tan concretos que no aparecen ni en problemas reales, asi que en vez de 
explorar por tipos de grafos, lo más adecuado sería explorar por grafos reales(pero me ocurre el problema ese de que
mi memoria no es suficiente)

'''

#SOLO SE HARÁN PRUEBAS CON SF
#RESETEO DE GRAFOS#--------------------------------------
subprocess.run(f"python central_generate_graph.py --num_graphs 15 --min_nodes 5000 --max_nodes 10000", shell=True,cwd="datasets")
subprocess.run(f"python central_create_dataset.py --split_train 5 --split_test 10 --model_size 10000 --num_copies 100" ,shell=True,cwd="datasets")
#Había que solo incluir SF, el resto TARDA MUCHO
#--------------------------------------------------------

gtypes=[]

#IDEA: Estudiar nº de capas óptima en GNN con grafos de entre 100000-50000 nodos(valores del artículo)
for i in range(1, 20):
    # Construir el comando con el valor de i
    print(f"###########BETWEENNESS CON {i} CAPAS INTERMEDIAS################")
    comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {i} --model_size 10000"
    comando_close=f"python closeness.py --g SF --num_intermediate_layer {i} --model_size 10000"
    comando_cluster=f"python local_clustering.py --g SF --num_intermediate_layer {i} --model_size 10000"
    comando_page=f"python page_rank.py --g SF --num_intermediate_layer {i} --model_size 10000"
    
    # Ejecutar el comando
    subprocess.run(comando_bet, shell=True)
    print(f"###########CLOSENESS CON {i} CAPAS INTERMEDIAS################")
    subprocess.run(comando_close, shell=True)
    print(f"########### LOCAL CLUSTERING CON {i} CAPAS INTERMEDIAS################")
    subprocess.run(comando_cluster, shell=True)
    print(f"###########AUTOVALOR CON {i} CAPAS INTERMEDIAS################")
    subprocess.run(comando_page, shell=True)
    print("-------------------------------------------------------------------------------------")
