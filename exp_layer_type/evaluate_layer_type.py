import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
random.seed(10)
import plotly.graph_objects as go

centrality="betweenness"

#TODO: ejecutar denuevo el de clustering porque se guardaba mal
types=["GNN","CNN","Transformer"]

def read_data(centrality,layers,gnn_type):
    data=[]
    for i in range(1,layers+1):
        with open(centrality+f"/{i}_{gnn_type}_kt.pickle","rb") as f:
            arrays=pickle.load(f)
            data.append(arrays[0][0])
            
    return data
    

dicc=dict()

for t in types:
    data=read_data(f"results/{centrality}",12,t)
    dicc.update({t:data})

fig = go.Figure()
# Agregar cada serie al gráfico con índices desde 1
for clave, valores in dicc.items():
    fig.add_trace(go.Scatter(
        x=[i+1 for i in range(len(valores))],  # Índices desde 1
        y=valores,
        mode='lines+markers',
        name=clave
    ))

# Configurar layout
fig.update_layout(
    title=f"Gráfica de {centrality}",
    xaxis_title="Número de capas intermedias",
    yaxis_title="KT última época",
    xaxis=dict(tickmode='linear', dtick=1)  # Asegurar que los ticks sean enteros
)

# Mostrar gráfica
fig.show()

