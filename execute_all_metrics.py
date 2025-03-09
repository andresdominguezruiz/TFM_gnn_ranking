import subprocess

# Rango de valores para i
for i in range(1, 16):
    # Construir el comando con el valor de i
    comando = f"python betweenness.py --g SF --num_intermediate_layer {i}"
    
    # Ejecutar el comando
    subprocess.run(comando, shell=True)
