from model.graph_matcher import GraphMatcher

import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # 3 GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class CodeSimilarityDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=4):
        super(CodeSimilarityDetectionModel, self).__init__()
        # GNN encoder for processing DFG
        self.gnn = GNNEncoder(input_dim, hidden_dim)
        
        # Graph matcher to compare two graphs
        self.matcher = GraphMatcher(
            psi_1=self.gnn,
            gembd_vec_dim=hidden_dim,
            aggregation='mean'
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,
                x_t, edge_index_t, edge_attr_t, batch_t):
        # Get graph embeddings from matcher
        embeddings, S_0 = self.matcher(
            x_s, edge_index_s, edge_attr_s, batch_s,
            x_t, edge_index_t, edge_attr_t, batch_t
        )
        
        # Classification
        logits = self.classifier(embeddings)
        return logits, S_0