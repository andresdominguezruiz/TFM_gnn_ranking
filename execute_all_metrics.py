import subprocess

# Rango de valores para i
for i in range(1, 16):
    # Construir el comando con el valor de i
    print(f"###########BETWEENNESS CON {i} CAPAS INTERMEDIAS################")
    comando_bet = f"python betweenness.py --g SF --num_intermediate_layer {i}"
    comando_close=f"python closeness.py --g SF --num_intermediate_layer {i}"
    comando_cluster=f"python local_clustering.py --g SF --num_intermediate_layer {i}"
    comando_page=f"python page_rank.py --g SF --num_intermediate_layer {i}"
    
    # Ejecutar el comando
    subprocess.run(comando_bet, shell=True)
    print(f"###########CLOSENESS CON {i} CAPAS INTERMEDIAS################")
    subprocess.run(comando_close, shell=True)
    print(f"########### LOCAL CLUSTERING CON {i} CAPAS INTERMEDIAS################")
    subprocess.run(comando_cluster, shell=True)
    print(f"###########AUTOVALOR CON {i} CAPAS INTERMEDIAS################")
    subprocess.run(comando_page, shell=True)
    print("-------------------------------------------------------------------------------------")
