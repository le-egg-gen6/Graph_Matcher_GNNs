import torch
from torch_geometric.data import Data

import torch
from torch_geometric.data import Data

def dfg_to_graph_data(dfg, code_tokens, code_token_embeds):
    """
    Convert a Data Flow Graph to PyTorch Geometric Data format.
    
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
    """
    # Create a mapping from token indices to node indices
    unique_nodes = set()
    for edge in dfg:
        code1, idx1, _, code2_list, idx2_list = edge
        unique_nodes.add(idx1)
        for idx2 in idx2_list:
            unique_nodes.add(idx2)
    
    node_indices = {idx: i for i, idx in enumerate(sorted(unique_nodes))}
    
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
            edge_attr.append(edge_types[edge_type])
    
    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    # Create node features
    if code_token_embeds is not None:
        # Use provided embeddings
        x = code_token_embeds[[node_indices[idx] for idx in sorted(unique_nodes)]]
    else:
        # Create one-hot encodings based on token vocabulary
        vocab = {token: idx for idx, token in enumerate(set(code_tokens))}
        num_nodes = len(node_indices)
        x = torch.zeros((num_nodes, len(vocab)), dtype=torch.float)
        for idx, node_idx in node_indices.items():
            token = code_tokens[idx]
            x[node_idx, vocab[token]] = 1.0
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(node_indices)
    )
    
    return data

def batch_dfg_to_pyg(dfgs, code_tokens_list, code_token_embeds_list=None):
    """
    Convert multiple DFGs to a batch of PyTorch Geometric Data objects.
    
    Args:
        dfgs (List[List[Tuple]]): List of DFGs
        code_tokens_list (List[List[str]]): List of code tokens for each DFG
        code_token_embeds_list (List[torch.Tensor], optional): List of pre-computed embeddings
    
    Returns:
        List[Data]: List of PyTorch Geometric Data objects
    """
    data_list = []
    
    for i, (dfg, tokens) in enumerate(zip(dfgs, code_tokens_list)):
        embeds = None if code_token_embeds_list is None else code_token_embeds_list[i]
        data = dfg_to_graph_data(dfg, tokens, embeds)
        data_list.append(data)
    
    return data_list

def create_graph_matcher_input(source_dfg, target_dfg, 
                             source_tokens, target_tokens,
                             source_embeds=None, target_embeds=None):
    """
    Create input format specifically for GraphMatcher model.
    
    Args:
        source_dfg: Source DFG
        target_dfg: Target DFG
        source_tokens: Source code tokens
        target_tokens: Target code tokens
        source_embeds: Optional pre-computed embeddings for source
        target_embeds: Optional pre-computed embeddings for target
    
    Returns:
        Tuple containing all required inputs for GraphMatcher:
        (x_s, edge_index_s, edge_attr_s, batch_s,
         x_t, edge_index_t, edge_attr_t, batch_t)
    """
    # Convert both DFGs to PyG format
    source_data = dfg_to_graph_data(source_dfg, source_tokens, source_embeds)
    target_data = dfg_to_graph_data(target_dfg, target_tokens, target_embeds)
    
    # Create batch indicators (all zeros for single graph)
    batch_s = torch.zeros(source_data.num_nodes, dtype=torch.long)
    batch_t = torch.zeros(target_data.num_nodes, dtype=torch.long)
    
    return (source_data.x, source_data.edge_index, source_data.edge_attr, batch_s,
            target_data.x, target_data.edge_index, target_data.edge_attr, batch_t)