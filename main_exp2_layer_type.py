import subprocess

'''
OBJETIVO DEL EXPERIMENTO:
- Estudiar comportamiento de los tipos de capas con diferentes tamaños,
para verificar lo aprendido en el exp1 (que no importa mucho la arquitectura del modelo,
sino los datos de entrenamiento)

RESULTADO: Para cada medida, tener una gráfica de rendimiento.

Evolución vista= con CNN, ya se nota el aprendizaje por cada época con diferentes números de capa.
Existe una relación directa con el nº de épocas y el nº de capas.

Cuanta menos capas utilices===> más épocas necesitaras para estabilizar el aprendizaje.



'''

#RESETEO DE GRAFOS#--------------------------------------

subprocess.run(f"python central_generate_graph.py --num_graphs 15 --min_nodes 3000 --max_nodes 5000", shell=True,cwd="datasets")
subprocess.run(f"python central_create_dataset.py --split_train 5 --split_test 10 --model_size 5000 --num_copies 100" ,shell=True,cwd="datasets")

#--------------------------------------------------------
#sage ya esta
types=["GAT","CNN","GNN"]
#IDEA: Estudiar comportamiento con diferentes capas utilizando menos nodos
for t in types:
    print(f"------------------------- TIPO DE GNN: {t} ---------------------------------------")
    for i in range(1, 10):
    # Construir el comando con el valor de i
        print(f"###########BETWEENNESS CON {i} CAPAS INTERMEDIAS################")
        comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {i} --model_size 5000 --gnn {t}"
        comando_close=f"python closeness.py --g SF --num_intermediate_layer {i} --model_size 5000 --gnn {t}"
        comando_cluster=f"python local_clustering.py --g SF --num_intermediate_layer {i} --model_size 5000 --gnn {t}"
        comando_page=f"python page_rank.py --g SF --num_intermediate_layer {i} --model_size 5000 --gnn {t}"
    
    # Ejecutar el comando
        if t == "GNN":
            subprocess.run(comando_bet, shell=True) #Asi solo se ejecuta GNN y SAGE 
        print(f"###########CLOSENESS CON {i} CAPAS INTERMEDIAS################")
        subprocess.run(comando_close, shell=True)
        print(f"########### LOCAL CLUSTERING CON {i} CAPAS INTERMEDIAS################")
        subprocess.run(comando_cluster, shell=True)
        print(f"###########AUTOVALOR CON {i} CAPAS INTERMEDIAS################")
        subprocess.run(comando_page, shell=True)
        print("-------------------------------------------------------------------------------------")
