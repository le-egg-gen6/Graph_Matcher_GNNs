#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: enhanced_graph_matcher.py
# Author: anon
# Email: nguyenle.workspace@gmail.com
# Created on: 2024-30-10
# 
# Distributed under terms of the MIT License

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import to_dense_batch, to_dense_adj
from .graph_matcher import GraphMatcher
from ..utils.processing_utils import (
    masked_softmax,
    reset,
    to_dense,
    to_sparse
)

__all__ = ['EnhancedGraphMatcher']

class EnhancedGraphMatcher(GraphMatcher):
    r"""
    Enhanced version of GraphMatcher with:
    - Cross-attention mechanism
    - Enhanced feature aggregation
    - Improved MLP for final embedding
    
    Args:
        psi_1: Primary GNN encoder
        psi_2: Secondary GNN encoder (optional)
        gembd_vec_dim: Dimension of graph embedding vector
        aggregation: Method to aggregate node features
    """
    def __init__(self, psi_1, psi_2=None, gembd_vec_dim=96, aggregation='attention'):
        super(EnhancedGraphMatcher, self).__init__(psi_1, psi_2, gembd_vec_dim, aggregation)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(gembd_vec_dim, 8, dropout=0.1)
        
        # Enhanced MLP with normalization and regularization
        self.mlp = nn.Sequential(
            nn.Linear(psi_1.out_channels * 4, psi_1.out_channels * 2),
            nn.LayerNorm(psi_1.out_channels * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(psi_1.out_channels * 2, gembd_vec_dim),
            nn.LayerNorm(gembd_vec_dim),
        )
        
    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,
                x_t, edge_index_t, edge_attr_t, batch_t, y=None):
        """
        Forward pass of the graph matcher
        
        Args:
            x_s, x_t: Source and target node features
            edge_index_s, edge_index_t: Source and target graph connectivity
            edge_attr_s, edge_attr_t: Source and target edge features
            batch_s, batch_t: Batch assignments for source and target nodes
            y: Ground truth matchings (optional)
            
        Returns:
            Graph embedding vector and similarity matrix
        """
        # Get node embeddings
        h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)
        
        # Convert to dense tensors
        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)
        
        # Apply cross-attention between graphs
        h_s_att, _ = self.cross_attention(h_s, h_t, h_t)
        h_t_att, _ = self.cross_attention(h_t, h_s, h_s)
        
        # Compute similarity scores
        S_hat = h_s @ h_t.transpose(-1, -2)
        S_mask = s_mask.view(h_s.size(0), h_s.size(1), 1) & t_mask.view(h_t.size(0), 1, h_t.size(1))
        S_0 = masked_softmax(S_hat, S_mask)
        
        # Aggregate features
        r_s = S_0 @ h_t
        r_t = S_0.transpose(-1, -2) @ h_s
        
        # Combine all features
        h_combined = torch.cat([h_s, r_s, h_s_att, h_t_att], dim=-1)
        h_combined = torch.mean(h_combined, dim=1)
        
        # Final embedding
        out = self.mlp(h_combined)
        
        return out, S_0

    def reset_parameters(self):
        self.psi_1.reset_parameters()
        if self.psi_2:
            self.psi_2.reset_parameters()
        self.reset(self.mlp)

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,
                        x_t, edge_index_t, edge_attr_t, batch_t, y=None):
        r"""
        Args:
            x_s (Tensor): Source graph node features of shape
                :param batch_t:
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_s (LongTensor): Source graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_s (Tensor): Source graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Source graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            x_t (Tensor): Target graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_t (LongTensor): Target graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_t (Tensor): Target graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Target graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            y (LongTensor, optional): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]` to include ground-truth values
                when training against sparse correspondences. Ground-truths
                are only used in case the model is in training mode.
                (default: :obj:`None`)

        Returns: A joint embedding vector for Gs, Gt of shape `[bsz * gembd_vec_dim]`
        """
        h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)
        h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)
        h_s, h_t = (h_s.detach(), h_t.detach()) if self.detach else (h_s, h_t)
        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)    # [64, 50, 48] or [bsz, Vt, dim]
        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'

        (B, N_s, C_out), N_t = h_s.size(), h_t.size(1)
        S_hat = h_s @ h_t.transpose(-1, -2)
        # Use S_hat to map any node func in L(Gs) -> L(Gt)
        S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
        S_0 = masked_softmax(S_hat, S_mask)  # [64, 17, 50]
        r_s = S_0 @ h_t
        h_st = torch.cat((h_s, r_s), dim=2)
        # --------------------------------------------------------- #
        if self.aggregation == 'mean':
            h_st = torch.mean(h_st, dim=1).squeeze()
        out = self.mlp(h_st)

        return out, S_0

    def reset(self, nn):
        def _reset(item):
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
        if nn is not None:
            if hasattr(nn, 'children') and len(list(nn.children())) > 0:
                for item in nn.children():
                    _reset(item)
            else:
                _reset(nn)

    def __repr__(self):
        return ('{}(\n'
                '    psi_1={},\n'
                '    psi_2={},\n'
                '    gembd_vec_dim={}\n').format(self.__class__.__name__,
                                                    self.psi_1, self.psi_2,
                                                    self.gembd_vec_dim)
