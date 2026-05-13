import torch
import torch.nn as nn
import torch.nn.functional as F


class ScatterAdd(nn.Module):
    """scatter_add_ wrapper - aggregates edge features by source node index."""
    def forward(self, src, index, dim_size):
        out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
        idx = index.unsqueeze(-1).expand(-1, src.size(-1))
        out.scatter_add_(0, idx, src)
        return out


class MEGNetBlock(nn.Module):
    """
    One MEGNet building block with edge and node update.

    Edge update:   e_ij' = phi_e([v_i, v_j, e_ij])          with residual
    Node update:   v_i'  = phi_v([v_i,  sum_j e_ij'])         with residual
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        # Edge update MLP
        self.edge_fc1 = nn.Linear(2 * node_dim + edge_dim, hidden_dim)
        self.edge_bn1 = nn.BatchNorm1d(hidden_dim)
        self.edge_fc2 = nn.Linear(hidden_dim, edge_dim)

        # Node update MLP
        self.node_fc1 = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.node_bn1 = nn.BatchNorm1d(hidden_dim)
        self.node_fc2 = nn.Linear(hidden_dim, node_dim)

        self.scatter = ScatterAdd()

    def forward(self, node_feat, edge_feat, edge_index):
        src, dst = edge_index  # [E], [E]

        # --- Edge update -----------------------------------------------
        edge_input = torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        h = self.edge_fc1(edge_input)
        if h.size(0) > 1:
            h = self.edge_bn1(h)
        h = F.relu(h)
        edge_update = self.edge_fc2(h)
        edge_feat = edge_feat + edge_update  # residual

        # --- Node update -----------------------------------------------
        agg = self.scatter(edge_feat, src, dim_size=node_feat.size(0))
        node_input = torch.cat([node_feat, agg], dim=-1)
        h = self.node_fc1(node_input)
        h = self.node_bn1(h)
        h = F.relu(h)
        node_update = self.node_fc2(h)
        node_feat = node_feat + node_update  # residual

        return node_feat, edge_feat


class MEGNet(nn.Module):
    """
    MEGNet for crystal property prediction (regression).

    Architecture:
      Embedding -> MEGNetBlock x n_blocks -> Mean pooling -> MLP -> output

    Parameters match CGCNN conventions so a fair comparison can be made
    on the same dataset and train/val/test split.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 node_dim=64, edge_dim=64, hidden_dim=128,
                 n_blocks=3, h_fea_len=128, n_h=1):
        super().__init__()
        # Initial embeddings
        self.node_embedding = nn.Linear(orig_atom_fea_len, node_dim)
        self.edge_embedding = nn.Linear(nbr_fea_len, edge_dim)

        # MEGNet blocks
        self.blocks = nn.ModuleList([
            MEGNetBlock(node_dim, edge_dim, hidden_dim)
            for _ in range(n_blocks)
        ])

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(node_dim, h_fea_len),
            nn.ReLU(),
        )

        # Hidden layers after pooling
        if n_h > 1:
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.hidden_acts = nn.ModuleList(
                [nn.ReLU() for _ in range(n_h - 1)]
            )
        else:
            self.hidden_layers = nn.ModuleList()
            self.hidden_acts = nn.ModuleList()

        self.fc_out = nn.Linear(h_fea_len, 1)

    def _build_edges(self, nbr_fea, nbr_fea_idx):
        """
        Convert CGCNN-style neighbor tensors into edge_index and edge features.

        nbr_fea:     [N, M, nbr_fea_len]   (M = max_num_nbr)
        nbr_fea_idx: [N, M]

        Returns (edge_index [2, E], edge_feat [E, edge_dim]).
        """
        N, M = nbr_fea_idx.shape
        dev = nbr_fea_idx.device

        src = torch.arange(N, device=dev).unsqueeze(1).expand(N, M).reshape(-1)
        dst = nbr_fea_idx.reshape(-1)
        nbr_fea_flat = nbr_fea.reshape(-1, nbr_fea.size(-1))

        # Filter out zero-padded edges: padded dst == 0 AND feature nearly zero
        valid = ~((dst == 0) & (nbr_fea_flat.abs().sum(dim=1) < 1e-6))
        # Also keep edges that point to valid atom indices
        valid = valid & (dst < N)

        src = src[valid]
        dst = dst[valid]
        fea = nbr_fea_flat[valid]
        return torch.stack([src, dst], dim=0), fea

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Parameters
        ----------
        atom_fea : [N, orig_atom_fea_len]   all atoms in the batch
        nbr_fea  : [N, M, nbr_fea_len]      bond features
        nbr_fea_idx : [N, M]                neighbor indices
        crystal_atom_idx : list of [n_i]    atom -> crystal mapping

        Returns [N0, 1] predictions for each crystal.
        """
        # Embed
        node_feat = self.node_embedding(atom_fea)  # [N, node_dim]

        # Build graph (already padded-edge filtered)
        edge_index, edge_raw = self._build_edges(nbr_fea, nbr_fea_idx)
        edge_feat = self.edge_embedding(edge_raw)  # [E, edge_dim]

        # MEGNet blocks
        for block in self.blocks:
            node_feat, edge_feat = block(node_feat, edge_feat, edge_index)

        # Mean pooling over atoms of each crystal
        crys_fea = torch.stack(
            [torch.mean(node_feat[idx], dim=0) for idx in crystal_atom_idx],
            dim=0,
        )  # [N0, node_dim]

        # Predict
        out = self.readout(crys_fea)  # [N0, h_fea_len]
        for fc, act in zip(self.hidden_layers, self.hidden_acts):
            out = act(fc(out))
        out = self.fc_out(out)  # [N0, 1]
        return out
