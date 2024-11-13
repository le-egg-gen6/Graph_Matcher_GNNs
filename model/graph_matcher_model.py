from .encoder import GraphEncoder
from .graph_matcher_core import GraphMatcher
from .tree_sitter_graph_builder import TreeSitterGraphBuilder


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Tuple, Dict, Optional, List
    

class CodeCloneDetection(nn.Module):
    """
    Model phát hiện code clone kết hợp TreeSitter, GraphEncoder và GraphMatcher.
    Có khả năng phát hiện các loại clone khác nhau:
    - Type-1: Giống hệt nhau (trừ whitespace, comments)
    - Type-2: Khác biệt về tên biến, kiểu dữ liệu
    - Type-3: Có thêm/bớt/sửa đổi statements
    - Type-4: Cùng chức năng nhưng khác cách implement
    
    Args:
        parsers: Dict[str, Parser] - TreeSitter parsers cho các ngôn ngữ
        hidden_channels: int - Số chiều của hidden layers
        out_channels: int - Số chiều của embeddings đầu ra
        num_layers: int - Số lớp GNN
        max_nodes: int - Số nodes tối đa trong một đồ thị
        dropout: float - Tỉ lệ dropout
        similarity_threshold: float - Ngưỡng để xác định code clone
    """
    def __init__(
        self,
        parsers,
        hidden_channels: int = 128,
        out_channels: int = 96,
        num_layers: int = 3,
        max_nodes: int = 5000,
        dropout: float = 0.1,
        similarity_threshold: float = 0.85,
        device: Optional[torch.device] = None
    ):
        super(CodeCloneDetection, self).__init__()

        # Set device
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Khởi tạo các components
        self.graph_builder = TreeSitterGraphBuilder(
            parsers=parsers,
            max_nodes=max_nodes,
            use_edge_types=True
        )
        
        # Graph encoder cho source code
        self.encoder = GraphEncoder(
            in_channels=self.graph_builder.node_type_count + 4,  # node type + extra features
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Graph matcher để so sánh các đồ thị
        self.matcher = GraphMatcher(
            psi_1=self.encoder,
            gembd_vec_dim=out_channels,
            aggregation='cat',
            detach=False
        )
        
        self.similarity_threshold = similarity_threshold
        
        # MLP để phân loại clone type
        self.clone_classifier = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 4)  # 4 clone types
        )

        # Move all components to the same device
        self.to(self._device)

    @property
    def device(self):
        return self._device
        
    def build_code_graphs(
        self,
        codes: List[str],
        lang: str,
        add_data_flow: bool = True,
        add_control_flow: bool = True
    ) -> Tuple[Batch, List[Data]]:
        """
        Xây dựng đồ thị cho một batch các đoạn code
        
        Returns:
            Tuple[Batch, List[Data]]: (Batched graph, List of individual graphs)
        """
        graphs = []
        for code in codes:
            graph = self.graph_builder.build_graph(
                code=code,
                lang=lang,
                add_data_flow=add_data_flow,
                add_control_flow=add_control_flow
            )
            graphs.append(graph)
            
        return Batch.from_data_list(graphs), graphs
        
    def forward(
        self,
        code1: str,
        code2: str,
        lang: str
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """
        So sánh hai đoạn code và phát hiện clone
        
        Returns:
            Tuple chứa:
            - Similarity score 
            - Attention matrix giữa các nodes
            - Clone type (0-3) hoặc None nếu không phải clone
        """
        # Build graphs
        batch_s, graphs_s = self.build_code_graphs([code1], lang)
        batch_t, graphs_t = self.build_code_graphs([code2], lang)

        # Move graphs to device
        batch_s = batch_s.to(self.device)
        batch_t = batch_t.to(self.device)
        
        # Get graph features
        graph_embedding, attention_matrix = self.matcher(
            x_s=batch_s.x,
            edge_index_s=batch_s.edge_index,
            edge_attr_s=batch_s.edge_attr,
            batch_s=batch_s.batch,
            x_t=batch_t.x, 
            edge_index_t=batch_t.edge_index,
            edge_attr_t=batch_t.edge_attr,
            batch_t=batch_t.batch
        )
        
        # Calculate similarity score
        similarity = F.cosine_similarity(
            graph_embedding[:self.matcher.gembd_vec_dim],
            graph_embedding[self.matcher.gembd_vec_dim:],
            dim=0
        )
        
        # Determine clone type if similar enough
        if similarity >= self.similarity_threshold:
            clone_logits = self.clone_classifier(graph_embedding)
            clone_type = torch.argmax(clone_logits).item()
            return similarity, attention_matrix, clone_type
        
        return similarity, attention_matrix, None
    
    def detect_clone_type(self, similarity: float, attention_matrix: torch.Tensor) -> int:
        """
        Phân tích ma trận attention và similarity để xác định loại clone
        
        Returns:
            int: Clone type (0-3)
        """
        # Type-1: Cấu trúc hoàn toàn giống nhau
        if similarity > 0.95:
            return 0
            
        # Type-2: Cấu trúc giống nhau nhưng khác tên biến
        attention_density = attention_matrix.sum() / (attention_matrix.size(0) * attention_matrix.size(1))
        if similarity > 0.9 and attention_density > 0.8:
            return 1
            
        # Type-3: Có sự thay đổi/thêm/bớt statements
        if similarity > 0.85:
            return 2
            
        # Type-4: Khác cấu trúc nhưng cùng chức năng
        if similarity > self.similarity_threshold:
            return 3
            
        return -1  # Không phải clone
        
    def explain_clone_detection(
        self,
        code1: str,
        code2: str,
        similarity: float,
        attention_matrix: torch.Tensor,
        clone_type: Optional[int]
    ) -> Dict:
        """
        Giải thích chi tiết kết quả phát hiện clone
        
        Returns:
            Dict chứa các thông tin giải thích
        """
        explanation = {
            "similarity": float(similarity),
            "is_clone": clone_type is not None,
            "clone_type": clone_type,
            "confidence": float(similarity) if clone_type is not None else 0.0,
            "attention_highlights": []
        }
        
        if clone_type is not None:
            # Analyze attention matrix to find matching regions
            matches = torch.nonzero(attention_matrix > 0.5)
            for i, j in matches:
                explanation["attention_highlights"].append({
                    "source_pos": int(i),
                    "target_pos": int(j),
                    "attention_weight": float(attention_matrix[i, j])
                })
                
        return explanation

    def reset_parameters(self):
        """Reset tất cả parameters về initialization values"""
        self.encoder.reset_parameters()
        self.matcher.reset_parameters()
        for layer in self.clone_classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()