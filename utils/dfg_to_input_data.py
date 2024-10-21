import torch
from torch_geometric.data import Data

def dfg_to_graph_data(dfg):
    # Bước 1: Tạo ánh xạ từ tên biến sang chỉ mục nút
    node_map = {}
    for edge in dfg:
        if edge[2] not in node_map:
            node_map[edge[2]] = len(node_map)
        if edge[4] not in node_map:
            node_map[edge[4]] = len(node_map)
    
    # Bước 2: Xây dựng tensor đặc trưng nút
    num_nodes = len(node_map)
    x = torch.zeros(num_nodes, 1)  # Đặc trưng đơn giản, can mở rộng
    
    # Bước 3: Tạo tensor chỉ mục cạnh
    edge_index = []
    for edge in dfg:
        source = node_map[edge[4]]
        target = node_map[edge[2]]
        edge_index.append([source, target])
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    # Bước 4: Tạo tensor đặc trưng cạnh (tùy chọn)
    edge_attr = torch.zeros(edge_index.size(1), 1)  # Đặc trưng đơn giản
    
    # Tạo đối tượng Data của PyTorch Geometric
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data