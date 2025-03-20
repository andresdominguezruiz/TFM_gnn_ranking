
import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
random.seed(10)
import plotly.graph_objects as go



layers=[1,3,5,7,10]

def read_data(layer):
    data=[]
    with open(f"results/betweenness/{layer}_GNN_SF_per_epoch_kt.pickle","rb") as f:
        list=pickle.load(f)
        for e in list:
            data.append(e[0])
    return data
            

    

dicc=dict()

for layer in layers:
    data=read_data(layer)
    dicc.update({f"{layer} Capas intermedias":data})

fig = go.Figure()
# Agregar cada serie al gráfico con índices desde 1
print(dicc)
for clave, valores in dicc.items():
    fig.add_trace(go.Scatter(
        x=[i+1 for i in range(len(valores))],  # Índices desde 1
        y=valores,
        mode='lines+markers',
        name=clave
    ))

# Configurar layout
fig.update_layout(
    title="Gráfica de evolución de las épocas",
    xaxis_title="Número de época",
    yaxis_title="KT obtenido",
    xaxis=dict(tickmode='linear', dtick=1)  # Asegurar que los ticks sean enteros
)

# Mostrar gráfica
fig.show()

