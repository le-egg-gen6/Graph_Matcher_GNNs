import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GlobalAttention
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, Dropout


class GraphEncoder(nn.Module):
    """
    Encoder hiệu quả cho GraphMatcher sử dụng kết hợp GCN và GAT
    với cơ chế attention toàn cục và skip connections.
    
    Args:
        in_channels (int): Số chiều của node features đầu vào
        hidden_channels (int): Số chiều của hidden layers
        out_channels (int): Số chiều của embeddings đầu ra
        num_layers (int): Số lớp GNN
        dropout (float): Tỉ lệ dropout
        use_batch_norm (bool): Có sử dụng batch normalization hay không
    """
    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        out_channels=96,
        num_layers=3,
        dropout=0.1,
        use_batch_norm=True
    ):
        super(GraphEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = Lin(in_channels, hidden_channels)
        
        # GNN Layers
        self.convs = nn.ModuleList()
        self.gats = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            # GCN layer for structure learning
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
            # GAT layer for attention-based message passing
            self.gats.append(GATConv(
                hidden_channels, 
                hidden_channels // 4, 
                heads=4, 
                dropout=dropout
            ))
            
            # Batch norm layer
            if use_batch_norm:
                self.bns.append(BatchNorm1d(hidden_channels))
        
        # Global attention pooling
        gate_nn = Seq(
            Lin(hidden_channels, hidden_channels // 2),
            ReLU(),
            Lin(hidden_channels // 2, 1)
        )
        self.global_attention = GlobalAttention(gate_nn)
        
        # Output projection
        self.output_proj = Seq(
            Lin(hidden_channels, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels),
            Dropout(dropout),
            Lin(hidden_channels, out_channels)
        )
        
        # Node feature augmentation
        self.node_feature_mlp = Seq(
            Lin(hidden_channels * 2, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass của encoder
        
        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_attr (Tensor, optional): Edge features [num_edges, num_edge_features]
            batch (LongTensor, optional): Batch vector [num_nodes]
            
        Returns:
            Tensor: Node embeddings [num_nodes, out_channels]
        """
        # Convert input features to float
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
            
        # Initial projection
        h = self.input_proj(x)
        
        # Store representations từ mỗi layer cho skip connections
        representations = []
        
        # Message passing layers
        for i in range(self.num_layers):
            # Combine GCN và GAT
            h1 = self.convs[i](h, edge_index)
            h2 = self.gats[i](h, edge_index)
            
            # Residual connection
            h = h + h1 + h2
            
            # Batch norm và nonlinearity
            if hasattr(self, 'bns'):
                h = self.bns[i](h)
            h = F.relu(h)
            
            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            representations.append(h)
        
        # Skip connections: kết hợp với representation từ layer đầu tiên
        h_combined = torch.cat([h, representations[0]], dim=-1)
        h = self.node_feature_mlp(h_combined)
        
        # Global attention pooling nếu cần
        if batch is not None:
            h_graph = self.global_attention(h, batch)
            # Broadcast graph-level features back to nodes
            h = h + h_graph[batch]
        
        # Output projection
        out = self.output_proj(h)
        
        return out
    
    def reset_parameters(self):
        """Reset tất cả parameters về initialization values"""
        for conv in self.convs:
            conv.reset_parameters()
        for gat in self.gats:
            gat.reset_parameters()
        if hasattr(self, 'bns'):
            for bn in self.bns:
                bn.reset_parameters()
                
        self.input_proj.reset_parameters()
        self.global_attention.reset_parameters()
        
        # Reset MLPs
        for layer in self.output_proj:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.node_feature_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()