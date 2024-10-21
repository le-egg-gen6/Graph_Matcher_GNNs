import torch
from torch_geometric.data import Data

def dfg_to_graph_matcher_format(dfg):
    """
    Convert a Data Flow Graph to the format expected by GraphMatcher.
    
    :param dfg: A dictionary representation of the DFG where keys are node IDs
                and values are dictionaries containing 'type' and 'edges'.
                'edges' is a list of target node IDs.
    :return: A PyTorch Geometric Data object
    """
    # Create a mapping of node types to integers
    node_types = set(node['type'] for node in dfg.values())
    type_to_idx = {t: i for i, t in enumerate(node_types)}
    
    # Create node features
    x = torch.zeros((len(dfg), len(type_to_idx)))
    for i, (_, node) in enumerate(dfg.items()):
        x[i, type_to_idx[node['type']]] = 1
    
    # Create edge index
    edge_index = []
    for source, node in dfg.items():
        for target in node['edges']:
            edge_index.append([int(source), int(target)])
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    # Create edge attributes (if needed)
    # For this example, we'll use a simple edge weight of 1 for all edges
    edge_attr = torch.ones(edge_index.size(1), 1)
    
    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data