import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer
from layer import GNN_Layer_Init
from layer import MLP
import torch
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GSAGE_Close(nn.Module):
    def __init__(self, ninput, nhid, dropout, num_intermediate_layers=6):
        super(GSAGE_Close, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput, nhid)
        self.intermediate_layers = nn.ModuleList(
            [geom_nn.SAGEConv(nhid, nhid) for _ in range(num_intermediate_layers)]
        )
        self.gc_last = geom_nn.SAGEConv(nhid, nhid)
        self.num_intermediate_layers = num_intermediate_layers

        self.dropout = dropout
        self.score_layer = MLP(nhid, self.dropout)

    def forward(self, adj1, adj2):
        device = next(self.parameters()).device
        adj1 = adj1.to(device)
        adj2 = adj2.to(device)
        
        x2_1 = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        x = x2_1
        for layer in self.intermediate_layers:
            x = F.normalize(F.relu(layer(x, adj2)), p=2, dim=1)
        x_last = F.relu(self.gc_last(x, adj2))

        scores = [self.score_layer(x2_1, self.dropout)]
        for layer in self.intermediate_layers:
            scores.append(self.score_layer(F.normalize(F.relu(layer(x2_1, adj2)), p=2, dim=1), self.dropout))
        scores.append(self.score_layer(x_last, self.dropout))

        score_top = sum(scores)

        return score_top

    def get_num_intermediate_layers(self):
        """Devuelve la cantidad de capas intermedias utilizadas en la red GNN."""
        return self.num_intermediate_layers

    def get_gnn_type(self):
        """Devuelve el tipo de GNN utilizado en la implementación."""
        return "SAGE"
