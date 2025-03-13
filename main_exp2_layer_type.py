import subprocess

#RESETEO DE GRAFOS#--------------------------------------

subprocess.run(f"python central_generate_graph.py --num_graphs 15 --min_nodes 4000 --max_nodes 7500", shell=True,cwd="datasets")
subprocess.run(f"python central_create_dataset.py --split_train 5 --split_test 10 --model_size 7500 --num_copies 100" ,shell=True,cwd="datasets")

#--------------------------------------------------------

types=["GNN","CNN","Transformer"]
#IDEA: Estudiar comportamiento con diferentes capas utilizando menos nodos
for t in types:
    print(f"------------------------- TIPO DE GNN: {t} ---------------------------------------")
    for i in range(1, 20):
    # Construir el comando con el valor de i
        print(f"###########BETWEENNESS CON {i} CAPAS INTERMEDIAS################")
        comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {i} --model_size 7500 --gnn {t}"
        comando_close=f"python closeness.py --g SF --num_intermediate_layer {i} --model_size 7500 --gnn {t}"
        comando_cluster=f"python local_clustering.py --g SF --num_intermediate_layer {i} --model_size 7500 --gnn {t}"
        comando_page=f"python page_rank.py --g SF --num_intermediate_layer {i} --model_size 7500 --gnn {t}"
    
    # Ejecutar el comando
        subprocess.run(comando_bet, shell=True)
        print(f"###########CLOSENESS CON {i} CAPAS INTERMEDIAS################")
        subprocess.run(comando_close, shell=True)
        print(f"########### LOCAL CLUSTERING CON {i} CAPAS INTERMEDIAS################")
        subprocess.run(comando_cluster, shell=True)
        print(f"###########AUTOVALOR CON {i} CAPAS INTERMEDIAS################")
        subprocess.run(comando_page, shell=True)
        print("-------------------------------------------------------------------------------------")
