from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d
from torch.nn.modules.container import ModuleList

from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATConv, LayerNorm

from layers import (
    CoAttentionLayer,
    RESCAL,
    IntraGraphAttention,
    InterGraphAttention, MyMLP, Pool, MultiHeadSelfAttention,
)

from data_preprocessing import ATOM_MAX_NUM

class DVRL(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, pooling_ratio, conv_channel1, conv_channel2, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)
        self.pooling_ratio = pooling_ratio
        self.max_num_nodes = ATOM_MAX_NUM
        self.dims = ceil(self.max_num_nodes * self.pooling_ratio)
        self.conv_channel1 = conv_channel1
        self.conv_channel2 = conv_channel2
        self.input_dim = 5392
        self.hidden_dims = [2048, 512]
        self.output_dim = self.conv_channel1 * self.conv_channel1

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = DVRL_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.conv1 = Conv2d(3, self.conv_channel1, (self.dims,1))
        self.maxpool2d = MaxPool2d((1, 2), (1, 2))
        self.conv2 = Conv2d(self.conv_channel1, self.conv_channel2, (1, self.hidd_dim //2), 1)

        self.co_attention = CoAttentionLayer(self.conv_channel1)
        self.mlp = MyMLP(self.input_dim, self.hidden_dims, self.output_dim)
        self.KGE = RESCAL(self.conv_channel1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, triples):
        h_data, t_data, cell, b_graph = triples

        batch_size = h_data.num_graphs

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-3)
        repr_t = torch.stack(repr_t, dim=-3)

        repr_h = F.relu(self.conv1(repr_h))
        repr_h = self.maxpool2d(repr_h)
        repr_h = F.relu(self.conv2(repr_h))
        repr_h = repr_h.view(batch_size, -1)
        repr_h = repr_h.unsqueeze(1)

        repr_t = F.relu(self.conv1(repr_t))
        repr_t = self.maxpool2d(repr_t)
        repr_t = F.relu(self.conv2(repr_t))
        repr_t = repr_t.view(batch_size, -1)
        repr_t = repr_t.unsqueeze(1)

        kge_heads = repr_h
        kge_tails = repr_t

        attentions = self.co_attention(kge_heads, kge_tails)
        cell = self.mlp(cell)
        scores = self.KGE(kge_heads, kge_tails, cell, attentions)

        return scores

class DVRL_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.pooling_ratio = 0.6
        self.max_num_nodes = ATOM_MAX_NUM
        self.dims = ceil(self.max_num_nodes * self.pooling_ratio)

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.intraAtt = IntraGraphAttention(head_out_feats*n_heads)
        self.interAtt = InterGraphAttention(head_out_feats*n_heads)
        self.pool = Pool(final_out_feats, ratio=self.pooling_ratio)
        self.att = MultiHeadSelfAttention(final_out_feats, final_out_feats, final_out_feats, 8)

    def reset_parameters(self):
        self.pool.reset_parameters()
        self.att.reset_parameters()

    def forward(self, h_data, t_data, b_graph):

        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep,t_interRep = self.interAtt(h_data,t_data,b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep

        h_pool_x, pool_edge_index, h_pool_batch = self.pool(h_data.x, h_data.edge_index, batch=h_data.batch)
        t_pool_x, pool_edge_index, t_pool_batch = self.pool(t_data.x, t_data.edge_index, batch=t_data.batch)

        h_batch_data, h_mask = to_dense_batch(h_pool_x, h_pool_batch, max_num_nodes=self.dims)
        t_batch_data, t_mask = to_dense_batch(t_pool_x, t_pool_batch, max_num_nodes=self.dims)

        h_att_x = self.att(h_batch_data)
        t_att_x = self.att(t_batch_data)

        return h_data, t_data, h_att_x, t_att_x

