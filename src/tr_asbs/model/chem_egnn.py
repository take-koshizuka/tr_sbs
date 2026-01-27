from typing import Tuple, Union

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from src.utils.graph_utils import subtract_com_batch

class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        n_atoms=10,
        hidden_nf=64,
        n_layers=4,
        agg="sum",
        uniform=False,
    ):
        super().__init__()
        in_node_nf = n_atoms + 1 if not uniform else 1
        in_edge_nf = 3 if not uniform else 1
        self.uniform = uniform
        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            agg=agg,
        )

    def forward(self, t, batch):
        if t.dim() == 0:
            n_systems = len(batch["ptr"]) - 1
            t = t * torch.ones(n_systems, device=batch["positions"].device)

        x = batch["positions"]
        edge_index = batch["edge_index"]
        batch_index = batch["batch"]
        h = torch.ones(x.shape[0], 1).to(x.device)
        h = h * t[batch_index, None]

        edge_attr = torch.sum(
            (x[edge_index[0]] - x[edge_index[1]]) ** 2, dim=1, keepdim=True
        )  # .double()
        if not self.uniform:
            h_atom_types = batch["node_attrs"]  # .double()
            h = torch.cat([h_atom_types, h], dim=-1)  # .double()
            bond_one_hot = torch.nn.functional.one_hot(batch["edge_attrs"][:, 1].long())
            edge_attr = torch.cat([bond_one_hot, edge_attr], dim=-1)
        x_final, _ = self.egnn(x, h, edge_index, edge_attr)
        return subtract_com_batch(x_final - x, batch_index)


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        n_layers=4,
        out_node_nf=None,
        # coords_range=15,
        agg="sum",
    ):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        # self.coords_range_layer = float(coords_range) / self.n_layers
        # if agg == "mean":
        #     self.coords_range_layer = self.coords_range_layer * 19
        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    in_edge_nf,
                    hidden_channels=self.hidden_nf,
                    aggr=agg,
                ),
            )

    def forward(self, x, h, edges, edge_attr):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            x, h = self._modules["gcl_%d" % i](x, h, edge_attr, edges)
        h = self.embedding_out(h)

        # # Important, the bias of the last linear might be non-zero
        # if node_mask is not None:
        #     h = h * node_mask
        return x, h


class E_GCL(MessagePassing):
    """EGNN layer from https://arxiv.org/pdf/2102.09844.pdf"""

    def __init__(
        self,
        channels_h: Union[int, Tuple[int, int]],
        channels_m: Union[int, Tuple[int, int]],
        channels_a: Union[int, Tuple[int, int]],
        aggr: str = "add",
        hidden_channels: int = 64,
        **kwargs,
    ):
        super(E_GCL, self).__init__(aggr=aggr, **kwargs)

        self.phi_e = nn.Sequential(
            nn.Linear(2 * channels_h + 1 + channels_a, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, channels_m),
            nn.LayerNorm(channels_m),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(channels_m, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.phi_h = ResWrapper(
            nn.Sequential(
                nn.Linear(channels_h + channels_m, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, channels_h),
            ),
            dim_res=channels_h,
        )

    def forward(self, x, h, edge_attr, edge_index, c=None):
        if c is None:
            c = degree(edge_index[0], x.shape[0]).unsqueeze(-1)
        return self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=edge_attr, c=c)

    def message(self, x_i, x_j, h_i, h_j, edge_attr):
        mh_ij = self.phi_e(
            torch.cat(
                [h_i, h_j, torch.norm(x_i - x_j, dim=-1, keepdim=True) ** 2, edge_attr],
                dim=-1,
            )
        )
        mx_ij = (x_i - x_j) * self.phi_x(mh_ij)
        return torch.cat((mx_ij, mh_ij), dim=-1)

    def update(self, aggr_out, x, h, edge_attr, c):
        m_len = 3
        m_x, m_h = aggr_out[:, :m_len], aggr_out[:, m_len:]
        h_l1 = self.phi_h(torch.cat([h, m_h], dim=-1))
        x_l1 = x + (m_x / c)
        return x_l1, h_l1


class ResWrapper(torch.nn.Module):
    def __init__(self, module, dim_res=2):
        super(ResWrapper, self).__init__()
        self.module = module
        self.dim_res = dim_res

    def forward(self, x):
        res = x[:, : self.dim_res]
        out = self.module(x)
        return out + res
