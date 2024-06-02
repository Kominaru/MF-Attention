import torch
from torch import nn
import numpy as np
import math
import random
from torch.nn import functional as F

limit_a, limit_b, epsilon = -.0, 1.1, 1e-6

class DynamicDataSelectionHard2(nn.Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""
    def __init__(self, n_features_to_select, temperature=2./3., start_dim=1, largest=True):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param temperature: Temperature of the concrete distribution
        """
        super(DynamicDataSelectionHard2, self).__init__()
        assert n_features_to_select > 0
        self.n_features_to_select = n_features_to_select
        self.temp = temperature
        self.start_dim = start_dim
        self.largest = largest
        self.coef = 1.
        self.factor_2 = .1

    def quantile_concrete(self, x, u):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""

        y = F.sigmoid(((torch.log(x) - torch.log(1 - x)) + u) / self.temp)

        return y

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.empty(size).uniform_(epsilon, 1-epsilon)
        eps = torch.autograd.Variable(eps)
        return eps

    def sample_z(self, u, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(torch.Size([batch_size, u.size(1)])).to(u.device)
            z = self.quantile_concrete(eps, u)
        else: 
            z = F.sigmoid(u/self.temp)

        x_shape = z.size()

        n_features = self.n_features_to_select if isinstance(self.n_features_to_select, int) else (
            int(np.prod(x_shape[1:]) * self.n_features_to_select))
        if self.n_features_to_select is None or (isinstance(self.n_features_to_select, float) and np.isclose(self.n_features_to_select, 1.)):
            mask = torch.ones_like(z)
        else:
            _, indices = torch.topk(z, n_features, dim=-1, largest=self.largest)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, indices, 1.)
            if self.factor_2 > 0 and sample:
                r = torch.rand((z.size(0), 1), device=mask.device)
                mask = torch.where(r < self.factor_2, torch.ones_like(mask), mask)

        s = z * (limit_b - limit_a) + limit_a

        return s, mask

    def forward(self, x, mask=None):
        x_shape = tuple(x.size())
        if len(x_shape) > 2:
            x = torch.flatten(x, start_dim=self.start_dim)

        x+=1.

        r, mask = self.sample_z(x, x.size(0), sample=self.training)

        s = F.hardtanh(r, min_val=0, max_val=1)

        if len(x_shape) > 2:
            mask = mask.view(x_shape)
            s = s.view(x_shape)
        return mask, s

class DynamicDataSelectionHard2v2(nn.Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""
    def __init__(self, n_features_to_select, weight_decay=1., droprate_init=0.5, temperature=5./3.,
                 lamba=1., local_rep=False, start_dim=1, largest=True, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(DynamicDataSelectionHard2v2, self).__init__()
        assert n_features_to_select > 0
        self.n_features_to_select = n_features_to_select
        self.temp = temperature
        self.start_dim = start_dim
        self.largest = largest
        self.factor = .1
        self.factor_2 = .1
 
    def quantile_concrete(self, u):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        r = 4. * torch.randn_like(u)
        y = torch.softmax((self.factor * r + u) / self.temp, dim=1)
        return y
 
    def sample_z(self, u, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            z = self.quantile_concrete(u)
        else:  # mode
            z = torch.softmax(u/self.temp, dim=1)
 
        x_shape = z.size()
 
        n_features = self.n_features_to_select if isinstance(self.n_features_to_select, int) else (
            int(np.prod(x_shape[1:]) * self.n_features_to_select))
 
        if self.n_features_to_select is None or (isinstance(self.n_features_to_select, float) and np.isclose(self.n_features_to_select, 1.)):
            mask = torch.ones_like(z)
        else:
            _, indices = torch.topk(z, n_features, dim=-1, largest=self.largest)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, indices, 1.)
            if self.factor_2 > 0 and sample:
                r = torch.rand((z.size(0), 1), device=mask.device)
                mask = torch.where(r < self.factor_2, torch.ones_like(mask), mask)
 
        s = n_features * z * (limit_b - limit_a) + limit_a
 
        return s, mask
 
    def forward(self, x, mask=None):
        x_shape = tuple(x.size())
        if len(x_shape) > 2:
            x = torch.flatten(x, start_dim=self.start_dim)
 
        r, mask= self.sample_z(x, x.size(0), sample=self.training)
 
        s = F.hardtanh(r, min_val=0, max_val=1)

        if len(x_shape) > 2:
            mask = mask.view(x_shape)
            s = s.view(x_shape)
        return mask, s