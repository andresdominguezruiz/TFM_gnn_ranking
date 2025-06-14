import torch.nn as nn
import torch.nn.functional as F
from layer import Transformer_Layer, GNN_Layer_Init, MLP
import torch
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#TODO: Cmabiar de Transformer a GAT
class GAT_Close(nn.Module):
    def __init__(self, ninput, nhid, dropout, num_intermediate_layers=6):
        super(GAT_Close, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput, nhid)
        self.intermediate_layers = nn.ModuleList([geom_nn.GATConv(nhid, nhid) for _ in range(num_intermediate_layers)])
        self.gc_last = geom_nn.GATConv(nhid, nhid)
        self.num_intermediate_layers = num_intermediate_layers

        self.dropout = dropout
        self.score_layer = MLP(nhid, self.dropout)

    def forward(self, adj1, adj2):
        device = next(self.parameters()).device
        adj1 = adj1.to(device)
        adj2 = adj2.to(device)
        x = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        for layer in self.intermediate_layers:
            x = F.normalize(F.relu(layer(x, adj2)), p=2, dim=1)
        x_last = F.relu(self.gc_last(x, adj2))

        scores = [self.score_layer(F.normalize(F.relu(layer(x, adj2)), p=2, dim=1), self.dropout) for layer in self.intermediate_layers]
        scores.insert(0, self.score_layer(x, self.dropout))
        scores.append(self.score_layer(x_last, self.dropout))

        score_top = sum(scores)
        return score_top
    
    def get_num_intermediate_layers(self):
        """Devuelve la cantidad de capas intermedias utilizadas en la red Transformer."""
        return self.num_intermediate_layers

    def get_gnn_type(self):
        """Devuelve el tipo de modelo utilizado en la implementación."""
        return "GAT"
