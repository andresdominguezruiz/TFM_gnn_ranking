import torch.nn as nn
import torch.nn.functional as F
from layer import CNN_Layer, GNN_Layer
from layer import GNN_Layer_Init
from layer import MLP
import torch 

class CNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout, num_intermediate_layers=4):
        super(CNN_Bet, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput, nhid)
        self.intermediate_layers = [CNN_Layer(nhid, nhid) for _ in range(num_intermediate_layers)]
        print(len(self.intermediate_layers))
        self.gc_last = CNN_Layer(nhid, nhid)
        self.num_intermediate_layers = num_intermediate_layers

        self.dropout = dropout
        self.score_layer = MLP(nhid, self.dropout)

    def forward(self, adj1, adj2):
        x1 = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        x2 = F.normalize(F.relu(self.gc1(adj2)), p=2, dim=1)

        for layer in self.intermediate_layers:
            x1 = F.normalize(F.relu(layer(x1, adj1)), p=2, dim=1)
            x2 = F.normalize(F.relu(layer(x2, adj2)), p=2, dim=1)

        x1_last = F.relu(self.gc_last(x1, adj1))
        x2_last = F.relu(self.gc_last(x2, adj2))

        scores1 = [self.score_layer(F.normalize(F.relu(layer(x1, adj1)), p=2, dim=1), self.dropout) for layer in self.intermediate_layers]
        scores2 = [self.score_layer(F.normalize(F.relu(layer(x2, adj2)), p=2, dim=1), self.dropout) for layer in self.intermediate_layers]

        scores1.insert(0, self.score_layer(x1, self.dropout))
        scores1.append(self.score_layer(x1_last, self.dropout))
        scores2.insert(0, self.score_layer(x2, self.dropout))
        scores2.append(self.score_layer(x2_last, self.dropout))

        score1 = sum(scores1)
        score2 = sum(scores2)

        return torch.mul(score1, score2)
    
    def get_num_intermediate_layers(self):
        """Devuelve la cantidad de capas intermedias utilizadas en la red GNN."""
        return self.num_intermediate_layers

    def get_gnn_type(self):
        """Devuelve el tipo de GNN utilizado en la implementación."""
        return "CNN"
