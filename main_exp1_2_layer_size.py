'''
OBJ: Estudiar comportamiento de la evolución de KT POR ÉPOCA de un tipo de modelo y grafo,
PERO CON DIFERENTES RESULTADOS. De esta forma, se puede estudiar la variabilidad.

RESULTADO: Gráfica similar a la de evolución por épocas, pero cada punto es la media de KT
obtenido de varias ejecuciones

'''

import subprocess


#SOLO SE HARÁN PRUEBAS CON SF
#RESETEO DE GRAFOS#--------------------------------------
subprocess.run(f"python central_generate_graph.py --num_graphs 15 --min_nodes 3000 --max_nodes 5000", shell=True,cwd="datasets")
subprocess.run(f"python central_create_dataset.py --split_train 5 --split_test 10 --model_size 5000 --num_copies 100" ,shell=True,cwd="datasets")
#Había que solo incluir SF, el resto TARDA MUCHO
#--------------------------------------------------------
num_capas=[1,3,5,7,10]
#IDEA: Estudiar nº de capas óptima en GNN con grafos de entre 100000-50000 nodos(valores del artículo)
for num in num_capas:
    
    # Construir el comando con el valor de i
    print(f"###########BETWEENNESS CON {num} CAPAS INTERMEDIAS, V1################")
    comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {num} --model_size 5000 --version V1"
    # Ejecutar el comando
    subprocess.run(comando_bet, shell=True)
    
    print(f"###########BETWEENNESS CON {num} CAPAS INTERMEDIAS, V2################")
    comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {num} --model_size 5000 --version V2"
    # Ejecutar el comando
    subprocess.run(comando_bet, shell=True)
    
    print(f"###########BETWEENNESS CON {num} CAPAS INTERMEDIAS, V3################")
    comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {num} --model_size 5000 --version V3"
    # Ejecutar el comando
    subprocess.run(comando_bet, shell=True)
    
    print(f"###########BETWEENNESS CON {num} CAPAS INTERMEDIAS, V4################")
    comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {num} --model_size 5000 --version V4"
    # Ejecutar el comando
    subprocess.run(comando_bet, shell=True)
    
    print(f"###########BETWEENNESS CON {num} CAPAS INTERMEDIAS, V5################")
    comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {num} --model_size 5000 --version V5"
    # Ejecutar el comando
    subprocess.run(comando_bet, shell=True)
    print("-------------------------------------------------------------------------------------")
