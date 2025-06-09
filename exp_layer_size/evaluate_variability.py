
import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
random.seed(10)
import plotly.graph_objects as go



layers=[1,3,5,7,10]
versions=["V1","V2","V3","V4","V5"]

def read_data(layer):
    data=[]
    list_v1=None
    list_v2=None
    list_v3=None
    list_v4=None
    list_v5=None
    with open(f"results/betweenness/{layer}_GNN_SF_per_epoch_V1_kt.pickle","rb") as f:
        list_v1=pickle.load(f)
    with open(f"results/betweenness/{layer}_GNN_SF_per_epoch_V2_kt.pickle","rb") as f:
        list_v2=pickle.load(f)
    with open(f"results/betweenness/{layer}_GNN_SF_per_epoch_V3_kt.pickle","rb") as f:
        list_v3=pickle.load(f)
    with open(f"results/betweenness/{layer}_GNN_SF_per_epoch_V4_kt.pickle","rb") as f:
        list_v4=pickle.load(f)
    with open(f"results/betweenness/{layer}_GNN_SF_per_epoch_V5_kt.pickle","rb") as f:
        list_v5=pickle.load(f)
    
    for e1,e2,e3,e4,e5 in zip(list_v1,list_v2,list_v3,list_v4,list_v5):
        aux=[e1[0],e2[0],e3[0],e4[0],e5[0]]
        data.append(np.mean(aux))
        
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
    title="Gráfica de evolución de las épocas con semillas diferentes",
    xaxis_title="Número de época",
    yaxis_title="Media de KT obtenido",
    xaxis=dict(tickmode='linear', dtick=1)  # Asegurar que los ticks sean enteros
)

# Mostrar gráfica
fig.show()

