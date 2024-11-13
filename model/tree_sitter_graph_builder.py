import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import List

class TreeSitterGraphBuilder(nn.Module):
    """
    Chuyển đổi AST từ tree-sitter thành graph cho GNN Encoder
    """
    def __init__(
        self,
        parsers,
        max_nodes=5000,
        use_edge_types=True
    ):
        super(TreeSitterGraphBuilder, self).__init__()
        self.parsers = parsers
        self.node_types = {}
        self.node_type_count = 512
        self.max_nodes = max_nodes
        self.edge_types = {
            "child": 0, 
            "next_sibling": 1, 
            "return_flow": 2, 
            "data_flow": 3
        }
        self.use_edge_types = use_edge_types

    def get_node_type_index(self, node_type: str) -> int:
        """Lấy hoặc tạo index cho node type"""
        if node_type not in self.node_types:
            self.node_types[node_type] = self.node_type_count
            self.node_type_count += 1
        return self.node_types[node_type]
    
    def extract_token_sequence(self, node, code: str) -> List[str]:
        """Trích xuất token sequence từ node"""
        if node.type == 'string':
            return ['STRING']
        elif node.type == 'number':
            return ['NUMBER']
        elif node.type == 'comment':
            return ['COMMENT']
        elif node.is_named:
            return [node.type]
        else:
            return [code[node.start_byte:node.end_byte]]
    def build_graph(
        self,
        code,
        lang,
        add_data_flow:True,
        add_control_flow:True
    ):
        """
        Xây dựng PyG graph từ code sử dụng tree-sitter AST
        
        Args:
            code: Source code string
            lang: Programming language
            add_data_flow: Whether to add data flow edges
            add_control_flow: Whether to add control flow edges
            
        Returns:
            PyG Data object containing the code graph
        """
        parser = self.parsers.get(lang)
        if not parser:
            raise ValueError(f"No parser found for language: {lang}")
            
        # Parse code
        tree = parser[0].parse(bytes(code, "utf8"))
        root = tree.root_node
        
        # Graph construction components
        node_features = []
        edge_index = []
        edge_attr = []
        
        # Mapping từ tree-sitter node về graph node index
        node_index_map = {}
        
        def process_node(node, parent_idx = None):
            if len(node_index_map) >= self.max_nodes:
                return
                
            current_idx = len(node_index_map)
            node_index_map[node.id] = current_idx
            
            # Node features
            node_type_idx = self.get_node_type_index(node.type)
            token_sequence = self.extract_token_sequence(node, code)
            
            # Combine node type và token features
            features = [
                node_type_idx,
                len(token_sequence),
                node.start_point[0],  # line number
                node.start_point[1],  # column
                1 if node.is_named else 0
            ]
            node_features.append(features)
            
            # Add edges
            if parent_idx is not None:
                # Child edge
                edge_index.append([parent_idx, current_idx])
                edge_attr.append([self.edge_types["child"]])
                
            # Process siblings
            prev_sibling_idx = None
            for child in node.children:
                if prev_sibling_idx is not None:
                    # Sibling edge
                    edge_index.append([prev_sibling_idx, current_idx])
                    edge_attr.append([self.edge_types["next_sibling"]])
                
                process_node(child, current_idx)
                prev_sibling_idx = current_idx
            
            return current_idx
            
        # Build basic AST structure
        process_node(root)
        
        # Add data flow edges if requested
        if add_data_flow:
            def find_var_nodes(node, vars_dict):
                if node.type in ['identifier', 'variable_declarator']:
                    name = code[node.start_byte:node.end_byte]
                    if name not in vars_dict:
                        vars_dict[name] = []
                    vars_dict[name].append(node_index_map[node.id])
                
                for child in node.children:
                    find_var_nodes(child, vars_dict)
            
            # Find all variable nodes
            variables = {}
            find_var_nodes(root, variables)
            
            # Add data flow edges
            for var_nodes in variables.values():
                for i in range(len(var_nodes) - 1):
                    edge_index.append([var_nodes[i], var_nodes[i + 1]])
                    edge_attr.append([self.edge_types["data_flow"]])
        
        # Add control flow edges if requested
        if add_control_flow:
            def process_control_flow(node, prev_idx=None):
                current_idx = node_index_map[node.id]
                
                if prev_idx is not None:
                    edge_index.append([prev_idx, current_idx])
                    edge_attr.append([self.edge_types["return_flow"]])
                
                if node.type in ['if_statement', 'while_statement', 'for_statement']:
                    # Connect condition to body
                    condition = node.child_by_field_name('condition')
                    body = node.child_by_field_name('body')
                    if condition and body:
                        edge_index.append([
                            node_index_map[condition.id],
                            node_index_map[body.id]
                        ])
                        edge_attr.append([self.edge_types["return_flow"]])
                
                for child in node.children:
                    process_control_flow(child, current_idx)
            
            process_control_flow(root)
        
        # Convert to PyG format
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr if self.use_edge_types else None,
            num_nodes=len(node_features)
        )
        
        return data