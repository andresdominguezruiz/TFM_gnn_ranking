import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
random.seed(10)
import plotly.graph_objects as go


#TODO: ejecutar denuevo el de clustering porque se guardaba mal
cents=["betweenness","closeness","page_rank","clustering"]

def read_data(centrality,layers):
    data=[]
    for i in range(1,layers+1):
        with open(centrality+f"/{i}_GNN_kt.pickle","rb") as f:
            arrays=pickle.load(f)
            data.append(arrays[0][0])
            
    return data
    

dicc=dict()

for cent in cents:
    data=read_data("results/"+cent,12)
    dicc.update({cent:data})

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
    title="Gráfica de Betweenness y Closeness",
    xaxis_title="Número de capas intermedias",
    yaxis_title="KT última época",
    xaxis=dict(tickmode='linear', dtick=1)  # Asegurar que los ticks sean enteros
)

# Mostrar gráfica
fig.show()

