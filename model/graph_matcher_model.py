from .code_to_graph import CodeToGraph
from .encoder import GNNEncoder
from .graph_matcher_core import GraphMatcher

import torch
import torch.nn as nn
from torch_geometric.nn import GlobalAttention
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class CodeSimilarityDetectionModel(nn.Module):
    def __init__(self, dfg_functions, model_name="microsoft/codebert-base", input_dim=768, hidden_dim=256, embedding_dim=128):
        super().__init__()
        self.code_to_dfg = CodeToGraph(dfg_functions, model_name)
        
        # Sequence-level encoder for Type-1 and Type-2 detection
        self.sequence_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.graph_encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            edge_dim=3
        )

        self.graph_attention = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ),
            nn=nn.Identity()
        )
        
        self.graph_matcher = GraphMatcher(self.graph_encoder, gembd_vec_dim=embedding_dim * 2)
        
        self.similarity_networks = nn.ModuleDict({
            'type1': nn.Linear(hidden_dim * 4, 1),
            'type2': nn.Linear(hidden_dim * 4, 1),
            'type3': nn.Linear(embedding_dim * 3, 1),
            'type4': nn.Linear(embedding_dim * 3, 1)
        })
        
        self.final_classifier = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4)
        )

    def encode_sequence_batch(self, token_embeddings_batch, lengths):
        """Encode sequence batch for Type-1 and Type-2 detection"""
        # Pack the padded sequence
        packed_embeddings = pack_padded_sequence(
            token_embeddings_batch, 
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process through LSTM
        _, (hidden, _) = self.sequence_encoder(packed_embeddings)
        
        # Concatenate bidirectional states
        batch_size = token_embeddings_batch.size(0)
        sequence_encoding = torch.cat([
            hidden[-2].view(batch_size, -1),
            hidden[-1].view(batch_size, -1)
        ], dim=1)
        
        return sequence_encoding

    def compute_sequence_similarity_batch(self, seq1_batch, seq2_batch):
        """Compute sequence-level similarity features for batches"""
        combined = torch.cat([seq1_batch, seq2_batch], dim=1)
        abs_diff = torch.abs(seq1_batch - seq2_batch)
        return combined, abs_diff
    
    def compute_graph_similarity_batch(self, graphs1, graphs2):
        """Compute graph-level similarity features for batches"""
        batch_size = len(graphs1)
        device = next(self.parameters()).device
        
        all_g1_pooled = []
        all_g2_pooled = []
        all_matching_scores = []
        
        for i in range(batch_size):
            # Process individual graphs
            g1_embed = self.graph_encoder(
                graphs1[i].x.to(device), 
                graphs1[i].edge_index.to(device), 
                graphs1[i].edge_attr.to(device)
            )
            g2_embed = self.graph_encoder(
                graphs2[i].x.to(device), 
                graphs2[i].edge_index.to(device), 
                graphs2[i].edge_attr.to(device)
            )
            
            # Pool graph embeddings first
            g1_batch = torch.zeros(g1_embed.size(0), dtype=torch.long, device=device)
            g2_batch = torch.zeros(g2_embed.size(0), dtype=torch.long, device=device)
            
            g1_pooled = self.graph_attention(g1_embed, g1_batch)
            g2_pooled = self.graph_attention(g2_embed, g2_batch)
            
            # Compute matching score with pooled embeddings
            matching_matrix = self.graph_matcher(
                g1_pooled, g2_pooled,
                graphs1[i].edge_index.to(device),
                graphs2[i].edge_index.to(device)
            )
            matching_score = matching_matrix.mean()  # Modified to handle pooled embeddings
            
            all_g1_pooled.append(g1_pooled)
            all_g2_pooled.append(g2_pooled)
            all_matching_scores.append(matching_score)
        
        return (
            torch.cat(all_g1_pooled, dim=0),
            torch.cat(all_g2_pooled, dim=0),
            torch.stack(all_matching_scores)
        )

    def forward(self, code1_batch, code2_batch, lang="java"):
        batch_size = len(code1_batch)
        device = next(self.parameters()).device
        
        # Convert codes to DFG graphs
        graphs1 = [self.code_to_dfg.convert(code1, lang) for code1 in code1_batch]
        graphs2 = [self.code_to_dfg.convert(code2, lang) for code2 in code2_batch]
        
        # Get lengths for padding
        lengths1 = torch.tensor([len(g.code_tokens) for g in graphs1])
        lengths2 = torch.tensor([len(g.code_tokens) for g in graphs2])
        
        # Pad token embeddings
        embeddings1 = pad_sequence([g.x for g in graphs1], batch_first=True)
        embeddings2 = pad_sequence([g.x for g in graphs2], batch_first=True)
        
        # Sequence-level encoding
        seq1_encoding = self.encode_sequence_batch(embeddings1.to(device), lengths1)
        seq2_encoding = self.encode_sequence_batch(embeddings2.to(device), lengths2)
        
        # Compute sequence similarities
        seq_combined, seq_diff = self.compute_sequence_similarity_batch(seq1_encoding, seq2_encoding)
        
        # Graph-level encoding
        g1_pooled, g2_pooled, matching_scores = self.compute_graph_similarity_batch(graphs1, graphs2)
        
        # Compute clone type probabilities
        type1_features = torch.cat([seq_combined, seq_diff], dim=1)
        type2_features = torch.cat([seq_combined, seq_diff], dim=1)
        
        type1_prob = torch.sigmoid(self.similarity_networks['type1'](type1_features))
        type2_prob = torch.sigmoid(self.similarity_networks['type2'](type2_features))
        
        # Graph features for Type-3 and Type-4
        graph_features = torch.cat([
            g1_pooled, g2_pooled,
            matching_scores.unsqueeze(1).to(device)
        ], dim=1)
        
        type3_prob = torch.sigmoid(self.similarity_networks['type3'](graph_features))
        type4_prob = torch.sigmoid(self.similarity_networks['type4'](graph_features))
        
        # Combine all probabilities
        all_probs = torch.cat([
            type1_prob, type2_prob, type3_prob, type4_prob
        ], dim=1)
        
        # Final classification
        clone_type_probs = self.final_classifier(all_probs)
        return torch.softmax(clone_type_probs, dim=1)
    
    def predict(self, code1_batch, code2_batch, lang="java"):
        self.eval()
        with torch.no_grad():
            probs = self(code1_batch, code2_batch, lang)
            return [{
                'type1': p[0].item(),
                'type2': p[1].item(),
                'type3': p[2].item(),
                'type4': p[3].item()
            } for p in probs]