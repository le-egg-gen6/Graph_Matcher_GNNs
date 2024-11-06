import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128, edge_dim=3):
        """
        Args:
            input_dim: Dimension of CodeBERT embeddings (default 768)
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output dimension for node embeddings
            edge_dim: Dimension of edge features
        """
        super().__init__()
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dimension reduction for CodeBERT embeddings
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.out_channels = output_dim
        self.attention = nn.Parameter(torch.randn(output_dim))
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)
        
    def forward(self, x, edge_index, edge_attr):
        # Project CodeBERT embeddings to hidden dimension
        x = self.input_projection(x)
        
        # Encode edge features
        edge_features = self.edge_encoder(edge_attr)
        
        # Graph convolutions with edge features and residual connections
        x1 = self.conv1(x, edge_index)
        x1 = x1 + torch.index_select(edge_features, 0, edge_index[0])
        x1 = self.layer_norm1(x1)
        x1 = torch.relu(x1 + x)
        
        x2 = self.conv2(x1, edge_index)
        x2 = x2 + torch.index_select(edge_features, 0, edge_index[0])
        x2 = self.layer_norm2(x2)
        x2 = torch.relu(x2 + x1)
        
        x3 = self.conv3(x2, edge_index)
        x3 = x3 + torch.index_select(edge_features, 0, edge_index[0])
        x3 = self.layer_norm3(x3)
        
        # Apply attention
        attention_weights = torch.softmax(torch.matmul(x3, self.attention), dim=0)
        x3 = x3 * attention_weights.unsqueeze(-1)
        
        return x3
    
    def reset_parameters(self):
        for layer in self.edge_encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.input_projection.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        nn.init.normal_(self.attention)