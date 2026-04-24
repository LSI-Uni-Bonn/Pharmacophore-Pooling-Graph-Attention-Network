import pandas as pd
import numpy as np
from rdkit import Chem
from reduceGraph import get_rg_edges_vectorized, get_rg_edges_with_mask, mol_to_graph, graph_to_pyg, mol_to_pool_idx, reduce_graph_from_mol
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import  GATv2Conv, global_mean_pool
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch.nn import Module
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_scatter import scatter_mean
from torch_geometric.data import Data



class GAT(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.edge_embed = Linear(edge_attr_dim, hidden_channels)

        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=hidden_channels, add_self_loops=False)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=hidden_channels,  add_self_loops=False)
        self.gat3 = GATv2Conv(hidden_channels, hidden_channels, heads=1, edge_dim=hidden_channels,  add_self_loops=False)
        self.gat4 = GATv2Conv(hidden_channels, hidden_channels, heads=1, edge_dim=hidden_channels,  add_self_loops=False)

        self.lin1 = Linear(hidden_channels, 64)
        self.lin2 = Linear(64, out_channels)

    def forward(self, data, edge_mask=None, return_attention=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # If explainer provides edge_mask, scale edge_attr with it
        #Instead of masking edges out of edge_index, apply the mask directly on edge_attr
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask.unsqueeze(-1)

        # Project edge attributes
        edge_attr = self.edge_embed(edge_attr)

        if return_attention:
            x, (edge_index, attn_weights1) = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index, edge_attr=edge_attr)


        #x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat3(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat4(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        if return_attention:
            return x, (edge_index, attn_weights1)
        else:
            return x

    
    def predict(self, data):
        self.eval()
        with torch.no_grad():
            data = data.to(next(self.parameters()).device)
            logits = self(data)
            probs = torch.sigmoid(logits)
            return probs.cpu()
    


class PPGAT(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels, out_channels, heads=4):
        super().__init__()

        # Edge attribute projection only for atom-level edges
        self.edge_embed = Linear(edge_attr_dim, hidden_channels)

        # Atom-level message passing (with edge_attr)
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=hidden_channels)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=hidden_channels)

        # Pharmacophore-level message passing (no edge_attr used)
        self.gat3 = GATv2Conv(hidden_channels, hidden_channels, heads=1)
        self.gat4 = GATv2Conv(hidden_channels, hidden_channels, heads=1)

          # Disable explaining for pharmacophore-level GATs
        self.gat3.explain = False
        self.gat4.explain = False

        # Fully connected layers
        self.lin1 = Linear(hidden_channels, 64)
        self.lin2 = Linear(64, out_channels)

    def forward(self, data, edge_mask=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # If explainer provides edge_mask, scale edge_attr with it
        #Instead of masking edges out of edge_index, apply the mask directly on edge_attr
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask.unsqueeze(-1)

        # Project atom-level edge attributes
        edge_attr = self.edge_embed(edge_attr)

        # Atom-level message passing
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # Graph reduction
        group_idx = data.pharma_index
        grouped_x = scatter_mean(x, index=group_idx, dim=0)
        reduced_batch = scatter_mean(batch.float(), index=group_idx, dim=0).long()
        #new_edge_index = data.new_edge_index
        #new_edge_attr = data.new_edge_attr

        # Get reduced graph edges dynamically
        
        #new_edge_index, new_edge_attr = get_rg_edges_vectorized(edge_index, group_idx)
        
      # Update reduced edges
        new_edge_index, new_edge_attr = get_rg_edges_with_mask(edge_index, group_idx, edge_mask) \
            if edge_mask is not None else get_rg_edges_vectorized(edge_index, group_idx)

        rg_data = Data(
            x=grouped_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,   
            batch=reduced_batch
        )


        if rg_data.edge_index.numel() > 0:
            x = self.gat3(rg_data.x, rg_data.edge_index)
            x = F.elu(x)
            x = self.gat4(x, rg_data.edge_index)
            x = F.elu(x)
        else:
            # If the explainer masked everything → skip message passing
             #handle case where new_edge_index is empty 
            x = rg_data.x



        # Global pooling and prediction
        x = global_mean_pool(x, rg_data.batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x

    def predict(self, data):
        self.eval()
        with torch.no_grad():
            data = data.to(next(self.parameters()).device)
            logits = self(data)
            probs = torch.sigmoid(logits)
            return probs.cpu()