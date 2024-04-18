import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from manifolds.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
from gnns import GraphConv



class HypFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.5, aggregate='add', args=None):
        super().__init__()
        self.manifold_in = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.manifold_out = Lorentz(k=float(args.k_out))
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act, args)
        self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn, gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate
        self.use_edge_loss = False
        self.gnn_use_bn = gnn_use_bn

        if self.aggregate == 'add':
            self.decode_trans = nn.Linear(self.hidden_channels + 1, self.out_channels)
            self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)
        elif self.aggregate == 'cat':
            self.decode_trans = nn.Linear(2*self.hidden_channels + 1, self.out_channels)
            self.decode_graph = nn.Linear(self.hidden_channels, self.out_channels)           
        else:
            raise ValueError(f'Invalid aggregate type:{self.aggregate}')

    def forward(self, x, edge_index):
        x1 = self.trans_conv(x)
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == 'add':
                x = (1-self.graph_weight)*self.decode_trans(x1) + self.graph_weight*self.decode_graph(x2)
            else:
                x = torch.cat((x1, x2), dim=-1)
                x = self.decode_trans(x)
        else:
            x = self.decode_trans(self.manifold_out.logmap0(x1)[...,1:])
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def reset_parameters(self):
        # self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
        # self.fc.reset_parameters()
        
class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = args.attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.power_k = args.power_k


    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p
    def linear_focus_attention(self, hyp_qs, hyp_ks, hyp_vs, output_attn=False):
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        v = hyp_vs[..., 1:]

        phi_qs = (F.relu(qs) + 1e-6) / self.norm_scale.abs()
        phi_ks = (F.relu(ks) + 1e-6) / self.norm_scale.abs()

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        # Step 1: Compute the kernel-transformed sum of K^T V across all N
        k_transpose_v = torch.einsum('nhd,ndh->hd', phi_ks, v)  # [H, D]

        # Step 2: Compute the kernel-transformed dot product of Q with the above result
        numerator = torch.einsum('nhd,hd->nh', phi_qs, k_transpose_v)  # [N, H]

        # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
        # denominator = torch.einsum('nhd->h', phi_ks)  # [H]
        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))  # [N, H]
        # Step 4: Normalize the numerator with the denominator
        # Note: Adding a small constant to the denominator for numerical stability
        attn_output = numerator / (denominator + 1e-6)

        v = self.v_map_mlp(v.mean(dim=1))
        # attn_output = attn_output + v.mean(dim=1)  # preserve its rank
        attn_output = attn_output + v  # preserve its rank
        attn_output = attn_output.unsqueeze(1)

        attn_output_time = ((attn_output**2).sum(dim=-1, keepdims=True) + self.manifold.k)**0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)
        attn_output = self.manifold.projx(attn_output)
        if output_attn:
            return attn_output, attn_output
        else:
            return attn_output

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)

        query = torch.stack(q_list, dim=1)  # [N, H, D]
        key = torch.stack(k_list, dim=1)  # [N, H, D]
        value = torch.stack(v_list, dim=1)  # [N, H, D]

        if output_attn:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(
                    query, key, value, output_attn)  # [N, H, D]
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(
                    query, key, value)  # [N, H, D]
            else:
                raise NotImplementedError


        final_output = attention_output
        # multi-head attention aggregation
        final_output = self.manifold.mid_point(final_output)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True, args=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.positional_embedding = HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden)

        self.bns = nn.ModuleList()
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))
        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels, num_heads=self.num_heads, use_weight=self.use_weight, args=args))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels, self.hidden_channels, self.manifold_out))

    def forward(self, x_input):
        layer_ = []
        x = self.fcs[0](x_input, x_manifold='euc')
        x_pos = self.positional_embedding(x_input, x_manifold='euc')
        x = self.manifold_hidden.mid_point(torch.stack((x, x_pos), dim=1))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = self.dropout(x, training=self.training)
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

