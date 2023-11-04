import torch
import torch.nn as nn

class MultiheadAttentionWithMimeticInit(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first,
                         device, dtype)
        self._reset_parameters1()

    # overriding the _reset_parameters function to initialize the projection weights using SVD
    def _reset_parameters1(self):
        # w: projection weights for q, k and v, packed into a single tensor. Weights
        #   are packed along dimension 0, in q, k, v order.

        assert self._qkv_same_embed_dim, 'Has to be the same for qkv'
        # super()._reset_parameters()

        # reinitlization of the projection weights using SVD
        Wq = self.in_proj_weight[:self.embed_dim, :]
        Wk = self.in_proj_weight[self.embed_dim:(2 * self.embed_dim), :]
        Wv = self.in_proj_weight[(2 * self.embed_dim):, :]

        Wout = self.out_proj.weight

        assert Wq.shape == Wk.shape == Wv.shape == Wout.shape, 'Shape of the projection weights has to be the same'

        # Mimetic initialization of the projection weights adapted from the paper https://arxiv.org/pdf/2305.09828.pdf
        alpha = 0.5
        beta = 1
        num_head = 1
        # for WqWk^T

        WqWk_t = alpha * torch.randn((Wq.shape[0], Wk.shape[0]), device=Wq.device) * 1 / num_head + beta * torch.eye(
            Wq.shape[0], Wk.shape[0], device=Wq.device)

        U, S, Vh = torch.linalg.svd(WqWk_t)
        S = torch.diag(S)
        Wv = torch.matmul(U, torch.sqrt(S))
        Wout = torch.matmul(torch.transpose(Vh, 0, 1), torch.sqrt(S))

        # for WqWv^T
        WvWout_t = alpha * torch.randn((Wv.shape[0], Wout.shape[0]),
                                       device=Wq.device) * 1 / num_head - beta * torch.eye(Wv.shape[0], Wout.shape[0],
                                                                                           device=Wq.device)
        U, S, Vh = torch.linalg.svd(WvWout_t)
        S = torch.diag(S)
        Wq = torch.matmul(U, torch.sqrt(S))
        Wk = torch.matmul(torch.transpose(Vh, 0, 1), torch.sqrt(S))

        self.in_proj_weight = nn.Parameter(torch.cat((Wq, Wk, Wv), dim=0)).requires_grad_(True)
        self.out_proj.weight = nn.Parameter(Wout).requires_grad_(True)

class Linear_Layers(nn.Module):
    def __init__(self, start, end, num_batch, activation=nn.LeakyReLU(), dropout=0.1, batchnorm=True, bias=True):
        super().__init__()
        self.infeature = start + num_batch
        self.outfeature = end
        self.num_batch = num_batch
        self.layers = nn.Linear(start + num_batch, end, bias=bias)
        self.bias = bias
        if bias == False:
            self.regularzation = nn.Parameter(torch.ones(start + num_batch))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(end)
        else:
            self.batchnorm = nn.Identity()

    def forward(self, x, batch):
        if self.num_batch != 0:
            x = torch.cat((x, batch), dim=1)

        if self.bias:
            x = self.layers(x)
        else:
            x = x * self.regularzation.reshape(1, self.infeature)
            linear_weight = self.layers.weight
            x = torch.nn.functional.linear(x, linear_weight)
        x = self.activation(x)
        # x = self.batchnorm(x)
        x = self.dropout(x)
        return x


class Linear_Layers_Classifier(nn.Module):
    def __init__(self, start, end, num_batch, activation=nn.LeakyReLU(), dropout=0.1, batchnorm=True, bias=True):
        super().__init__()
        self.infeature = start + num_batch
        self.outfeature = end
        self.num_batch = num_batch
        self.layers = nn.Linear(start + num_batch, end, bias=bias)
        self.bias = bias
        if bias == False:
            self.regularzation = nn.Parameter(torch.ones(start + num_batch))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if batchnorm:
            self.layernorm = nn.LayerNorm(self.infeature)
        else:
            self.layernorm = nn.Identity()

    def forward(self, x, batch):
        if self.num_batch != 0:
            x = torch.cat((x, batch), dim=1)
        # x = self.layernorm(x)
        if self.bias:
            x = self.layers(x)
        else:
            x = x * self.regularzation.reshape(1, self.infeature)
            linear_weight = self.layers.weight
            x = torch.nn.functional.linear(x, linear_weight)
        x = self.activation(x)
        x = self.dropout(x)
        return x

