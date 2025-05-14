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


MODEL="CNN"
TASK="clustering"

def read_data(layer):
    data_mean_0 = []
    data_std_1 = []
    
    # Leer ficheros
    list_v1 = pickle.load(open(f"results/{TASK}/{layer}_{MODEL}_SF_per_epoch_V1_kt.pickle","rb"))
    list_v2 = pickle.load(open(f"results/{TASK}/{layer}_{MODEL}_SF_per_epoch_V2_kt.pickle","rb"))
    list_v3 = pickle.load(open(f"results/{TASK}/{layer}_{MODEL}_SF_per_epoch_V3_kt.pickle","rb"))
    list_v4 = pickle.load(open(f"results/{TASK}/{layer}_{MODEL}_SF_per_epoch_V4_kt.pickle","rb"))
    list_v5 = pickle.load(open(f"results/{TASK}/{layer}_{MODEL}_SF_per_epoch_V5_kt.pickle","rb"))
    
    # Procesar por época
    for e1, e2, e3, e4, e5 in zip(list_v1, list_v2, list_v3, list_v4, list_v5):
        aux_0 = [e1[0], e2[0], e3[0], e4[0], e5[0]]
        aux_1 = [e1[1], e2[1], e3[1], e4[1], e5[1]]
        
        data_mean_0.append(np.mean(aux_0))
        data_std_1.append(np.std(aux_1))  # Desviación típica
        
    return data_mean_0, data_std_1
            

dicc_medias = dict()
dicc_std = dict()

for p in layers:
    medias, desvios = read_data(p)
    dicc_medias[f"{TASK}-{MODEL}-layers={p}"] = medias
    dicc_std[f"{TASK}-{MODEL}-layers={p}"] = desvios

fig = go.Figure()

for clave in dicc_medias.keys():
    fig.add_trace(go.Scatter(
        x=[i+1 for i in range(len(dicc_medias[clave]))],
        y=dicc_medias[clave],
        mode='lines+markers',
        name=clave,
        error_y=dict(
            type='data',
            array=dicc_std[clave],
            visible=True
        )
    ))

fig.update_layout(
    title=f"{TASK}-{MODEL}-Gráfica para la evolución de los modelos según cantidad de capas",
    xaxis_title="Número de época",
    yaxis_title="Media de KT obtenido",
    xaxis=dict(tickmode='linear', dtick=1)
)

fig.show()

# RECUERDA QUE LA INTENCIÓN DE ESTA EXPERIMENTACIÓN ES VER SI OCURRE LO QUE PREEVÍA CON
#page rank y clustering=> a mayor nº de capas, mayor nº de épocas necesita para estabilizarse
#, Y ASI OCURRE CON PAGE_RANK
