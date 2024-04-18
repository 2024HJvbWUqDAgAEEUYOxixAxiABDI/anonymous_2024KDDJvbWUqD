import pdb

import torch.nn as nn
import torch.nn.functional
import torch.nn.init as init
from manifolds.lorentz import Lorentz
import math
from geoopt import ManifoldParameter
from geoopt.optim.rsgd import RiemannianSGD
from geoopt.optim.radam import RiemannianAdam



class HypLayerNorm(nn.Module):
    def __init__(self, manifold, in_features, manifold_out=None):
        super(HypLayerNorm, self).__init__()
        self.in_features = in_features
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.layer(x_space)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class HypNormalization(nn.Module):
    def __init__(self, manifold, manifold_out=None):
        super(HypNormalization, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class HypActivation(nn.Module):
    def __init__(self, manifold, activation, manifold_out=None):
        super(HypActivation, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.activation = activation

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = self.activation(x_space)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x
class HypDropout(nn.Module):
    def __init__(self, manifold, dropout, manifold_out=None):
        super(HypDropout, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=False):
        if training:
            x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
            if self.manifold_out is not None:
                x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypLinear(nn.Module):
    """
    Parameters:
        manifold (manifold): The manifold to use for the linear transformation.
        in_features (int): The size of each input sample.
        out_features (int): The size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        dropout (float, optional): The dropout probability. Default is 0.1.
    """

    def __init__(self, manifold, in_features, out_features, bias=True, dropout=0.0, manifold_out=None):
        super().__init__()
        self.in_features = in_features + 1  # + 1 for time dimension
        self.out_features = out_features
        self.bias = bias
        self.manifold = manifold
        self.manifold_out = manifold_out
        
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.dropout_rate = dropout
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp'):
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        x_space = self.linear(x)
        # x_space = torch.nn.functional.dropout(x_space, p=self.dropout_rate, training=self.training)

        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class Optimizer(object):
    def __init__(self, model, args):
        euc_optimizer_type = args.euc_optimizer_type
        hyp_optimizer_type = args.hyp_optimizer_type
        euc_lr = args.lr
        hyp_lr = args.hyp_lr
        euc_weight_decay = args.weight_decay
        hyp_weight_decay = args.hyp_weight_decay

        euc_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and not isinstance(p, ManifoldParameter)]

        hyp_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and isinstance(p, ManifoldParameter)]

        if euc_optimizer_type == 'adam':
            optimizer_euc = torch.optim.Adam(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)

        elif euc_optimizer_type == 'sgd':
            optimizer_euc = torch.optim.SGD(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)

        else:
            raise NotImplementedError

        # if hyp_optimizer_type == 'radam':
        #     optimizer_hyp = RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=10, weight_decay=hyp_weight_decay)

        # elif hyp_optimizer_type == 'rsgd':
        #     optimizer_hyp = RiemannianSGD(hyp_params, lr=hyp_lr, stabilize=10, weight_decay=hyp_weight_decay)
        # else:
        #     raise NotImplementedError

        # self.optimizer = [optimizer_euc, optimizer_hyp]
        self.optimizer = [optimizer_euc]


    def step(self):
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizer:
            optimizer.zero_grad()
