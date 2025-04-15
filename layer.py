import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F



class GNN_Layer(Module):
    """
    Layer defined for GNN-Bet
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        device="cuda:0" if torch.cuda.is_available() else "cpu"
        support = torch.mm(input, self.weight.to(device))
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias.to(device)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class GNN_Layer_Init(Module):
    """
    First layer of GNN_Init, for embedding lookup
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer_Init, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        support = self.weight
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLP(Module):
    def __init__(self, nhid,dropout):
        super(MLP,self).__init__()
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(nhid,2*nhid)
        self.linear2 = torch.nn.Linear(2*nhid,2*nhid)
        self.linear3 = torch.nn.Linear(2*nhid,1)


    def forward(self,input_vec,dropout):

        score_temp = F.relu(self.linear1(input_vec))
        score_temp = F.dropout(score_temp,self.dropout,self.training)
        score_temp = F.relu(self.linear2(score_temp))
        score_temp = F.dropout(score_temp,self.dropout,self.training)
        score_temp = self.linear3(score_temp)

        return score_temp

###TIPOS DE CAPA PARA EXPERIMENTACIONES###----------------------------------------

class CNN_Layer(Module):
    """
    GNN Layer with CNN applied to the adjacency matrix
    """

    def __init__(self, in_features, out_features, bias=True):
        super(CNN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Aplicamos convolución 2D sobre la matriz de adyacencia
        self.conv_adj = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        # Parámetro de transformación de características
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        input: (batch_size, num_nodes, in_features) -> características de nodos
        La idea de las capas convolucionales es que están hechas para procesamiento de imágenes,
        por lo que la matriz de adyacencia hay que TRANSFORMARLA en algo con las mismas dimensiones que una imagen.
        adj: (batch_size, num_nodes, num_nodes) o (num_nodes, num_nodes)
        """
        
        if adj.is_sparse:
            adj = adj.to_dense()  # Convierte a FloatTensor denso

        # Asegurar que adj sea un tensor de 4 dimensiones: (batch_size, 1, num_nodes, num_nodes)
        if adj.dim() == 2:  # Si la entrada es (num_nodes, num_nodes), añadir batch y canal
            adj = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, num_nodes, num_nodes)
        elif adj.dim() == 3:  # Si la entrada es (batch_size, num_nodes, num_nodes)
            adj = adj.unsqueeze(1)  # (batch_size, 1, num_nodes, num_nodes)

        # Aplicamos la convolución sobre la matriz de adyacencia
        adj = self.conv_adj(adj)  # (batch_size, 1, num_nodes, num_nodes)

        # Volvemos a (batch_size, num_nodes, num_nodes) eliminando el canal
        adj = adj.squeeze(1)  

        # Propagación de características con la adyacencia convolucionada
        support = torch.matmul(input, self.weight)  # (batch_size, num_nodes, out_features)
        output = torch.matmul(adj, support)  # (batch_size, num_nodes, out_features)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class Transformer_Layer(Module):
    """
    Transformer Layer for adjacency matrices (replacing GNN)
    """
    
    def __init__(self, in_features, out_features, bias=True, num_heads=4, dropout=0.1):
        super(Transformer_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # Weight matrix equivalent
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        num_heads = max(1, out_features // 64)  # Asegura que sea un valor válido
        num_heads = min(num_heads, out_features)  # Evita que sea mayor que `out_features`
        self.attention = torch.nn.MultiheadAttention(embed_dim=out_features, num_heads=num_heads, dropout=dropout, batch_first=True)
        # Optional bias term
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes weights using Xavier initialization.
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        """
        input: (batch_size, num_nodes, in_features) - Node features
        adj: (batch_size, num_nodes, num_nodes) - Adjacency matrix
        """
        support = torch.matmul(input, self.weight)  # (batch_size, num_nodes, out_features)
        attn_output, _ = self.attention(support, support, support)  # Self-attention
        
        if self.bias is not None:
            return attn_output + self.bias
        else:
            return attn_output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features}, heads={self.num_heads})"
