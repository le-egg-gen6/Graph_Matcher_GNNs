import torch
from torch_geometric.data import Data

import torch
from torch_geometric.data import Data

def dfg_to_graph_data(dfg, code_tokens, code_token_embeds=None):
    """
    Convert a Data Flow Graph to PyTorch Geometric Data format with code tokens.
    
    Args:
        dfg (List[Tuple]): List of DFG edges in format (code1, idx1, edge_type, [code2], [idx2])
        code_tokens (List[str]): List of code tokens
        code_token_embeds (torch.Tensor, optional): Pre-computed embeddings for code tokens.
            Shape should be [num_nodes, embedding_dim]. If None, will create one-hot encodings.
    
    Returns:
        Data: A PyTorch Geometric Data object with the following attributes:
            - x: Node features
            - edge_index: Graph connectivity in COO format
            - edge_attr: Edge features/types
            - code_tokens: Original code tokens
            - node_indices: Mapping from original indices to graph indices
            - original_indices: Mapping from graph indices to original indices
    """
    # Create a mapping from token indices to node indices
    unique_nodes = set()
    for edge in dfg:
        code1, idx1, _, code2_list, idx2_list = edge
        unique_nodes.add(idx1)
        for idx2 in idx2_list:
            unique_nodes.add(idx2)
    
    node_indices = {idx: i for i, idx in enumerate(sorted(unique_nodes))}
    original_indices = {i: idx for idx, i in node_indices.items()}
    
    # Create edge index and edge attributes
    edge_index = []
    edge_attr = []
    edge_types = {'comesFrom': 0, 'computedFrom': 1}
    
    for edge in dfg:
        code1, idx1, edge_type, code2_list, idx2_list = edge
        src_idx = node_indices[idx1]
        
        for idx2 in idx2_list:
            dst_idx = node_indices[idx2]
            edge_index.append([dst_idx, src_idx])  # Note: reversed for message passing
            edge_attr.append(edge_types.get(edge_type, 0))  # Default to 0 if edge_type not found
    
    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(-1) if edge_attr else torch.zeros((0, 1), dtype=torch.float)
    
    # Create node features
    if code_token_embeds is not None:
        # Use provided embeddings for the nodes we have
        x = code_token_embeds
    else:
        # Create one-hot encodings based on token vocabulary
        vocab = {token: idx for idx, token in enumerate(set(code_tokens))}
        num_nodes = len(code_tokens)
        x = torch.zeros((num_nodes, len(vocab)), dtype=torch.float)
        for idx, token in enumerate(code_tokens):
            x[idx, vocab[token]] = 1.0
    
    # Store the original code tokens and create mask for valid nodes
    valid_node_mask = torch.zeros(len(code_tokens), dtype=torch.bool)
    for orig_idx in unique_nodes:
        if orig_idx < len(code_tokens):  # Safety check
            valid_node_mask[orig_idx] = True
    
    # Create PyG Data object with all necessary attributes
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(code_tokens),
        code_tokens=code_tokens,  # Store original tokens
        valid_node_mask=valid_node_mask,  # Mask for valid nodes in the graph
        node_indices=node_indices,  # Mapping from original indices to graph indices
        original_indices=original_indices  # Mapping from graph indices to original indices
    )
    
    return data