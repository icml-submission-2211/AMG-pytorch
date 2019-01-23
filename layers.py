import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_dim, hidden_dim, num_classes, n_hop, act, dropout, bias=True):
        super(GraphConvolution, self).__init__()

        self.n_hop = n_hop # 6/4/3
        self.alpha = nn.Parameter(torch.tensor(0.6, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(0.4, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(0.3, requires_grad=True))

        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.u_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim))
        #self.v_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim))
        self.v_weight = self.u_weight
        if bias:
            self.u_bias = Parameter(torch.randn(hidden_dim))
        else:
            self.u_bias = None

        self.out_u = nn.Linear(hidden_dim, hidden_dim)
        self.out_v = nn.Linear(hidden_dim, hidden_dim)

        for w in [self.u_weight, self.v_weight, self.out_u.weight, self.out_v.weight]:
            nn.init.xavier_normal_(w)

    def n_square(self, A):
        mx = torch.cat((torch.cat((torch.zeros(A.size(0),A.size(1),A.size(1)).cuda(), A), 2),
                        torch.cat((A.permute(0,2,1), torch.zeros(A.size(0),A.size(2),A.size(2)).cuda()), 2)), 1)

        square = torch.eye(A.size(1)+A.size(2)).expand(A.size(0),-1,-1).cuda()
        single = 0.2 * square.detach() + 0.8 * self.normalize(mx)
        for i in range(self.n_hop):
            square = torch.einsum('rku,ruv->rkv', square, single)

        return square[:,:A.size(1),A.size(1):], square[:,A.size(1):,:A.size(1)]

    def normalize(self, adjacencies):
        adj_tot = torch.sum(adjacencies, 0)
        degree_u = torch.sum(adj_tot, 1)
        degree_v = torch.sum(adj_tot, 0)

        degree_u_inv_sqrt = 1. / torch.sqrt(degree_u)
        degree_v_inv_sqrt = 1. / torch.sqrt(degree_v)
        # set zeros to inf to avoid dividing by zero
        degree_u_inv_sqrt[torch.isinf(degree_u_inv_sqrt)] = 0.
        degree_v_inv_sqrt[torch.isinf(degree_v_inv_sqrt)] = 0.
        degree_u_inv_sqrt_mat = torch.diag(degree_u_inv_sqrt)
        degree_v_inv_sqrt_mat = torch.diag(degree_v_inv_sqrt)

        adj_norm = [degree_u_inv_sqrt_mat.matmul(adj).matmul(degree_v_inv_sqrt_mat) for adj in adjacencies]
        adj_norm = torch.stack(adj_norm, 0)

        return adj_norm

    def n_sum(self, A):
        mx = torch.cat((torch.cat((torch.zeros(A.size(0),A.size(1),A.size(1)).to(A.device), A), 2),
                        torch.cat((A.permute(0,2,1), torch.zeros(A.size(0),A.size(2),A.size(2)).to(A.device)), 2)), 1)

        eye = torch.eye(A.size(1)+A.size(2)).expand(A.size(0),-1,-1).to(A.device)
        #single = 0.2 * eye.detach() + 0.8 * self.normalize(mx)
        single = self.normalize(mx)

        square = torch.matmul(eye.detach(), mx)
        square = torch.matmul(square, mx)

        cube = torch.matmul(eye.detach(), mx)
        cube = torch.matmul(cube, mx)
        cube = torch.matmul(cube, mx)
        cube = cube + cube.permute(0,2,1)

        quad = torch.matmul(eye.detach(), mx)
        quad = torch.matmul(quad, mx)
        quad = torch.matmul(quad, mx)
        quad = torch.matmul(quad, mx)

        alpha = (1.+torch.tanh(self.alpha))*0.5
        beta = (1.+torch.tanh(self.beta))*0.5*(1.-alpha)
        gamma = (1.+torch.tanh(self.gamma))*0.5*(1.-alpha-beta)
        delta = 1.-alpha-beta-gamma

        return alpha * single + beta * self.normalize(square) + gamma * self.normalize(cube) + delta * self.normalize(quad)

    def forward(self, u_feat, v_feat, support):

        u_feat = self.dropout(u_feat)
        v_feat = self.dropout(v_feat)
        feat = torch.cat((u_feat, v_feat), 0)
        supports = []
        weight = 0
        support_norm = self.n_sum(support)
        for r in range(support.size(0)):
            weight = weight + self.u_weight[r]

            # multiply feature matrices with weights
            tmp = torch.mm(feat, weight)

            # then multiply with rating matrices
            supports.append(torch.mm(support_norm[r], tmp))

        z = torch.sum(torch.stack(supports, 0), 0)

        if self.u_bias is not None:
            z = z + self.u_bias

        outputs = self.act(z)

        output_u = self.act(self.out_u(outputs[:u_feat.size(0)]))
        output_v = self.act(self.out_v(outputs[u_feat.size(0):]))
        return output_u, output_v


class BilinearMixture(Module):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """
    def __init__(self, num_users, num_items, num_classes, input_dim,
                 nb=5, dropout=0.7, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)

        self.dropout = nn.Dropout(dropout)
        self.weight = Parameter(torch.randn(num_classes, input_dim, input_dim))
        #self.a = Parameter(torch.randn(nb, num_classes))

        self.u_bias = Parameter(torch.randn(num_users, num_classes))
        self.v_bias = Parameter(torch.randn(num_items, num_classes))

        for w in [self.weight, self.u_bias, self.v_bias]:
            nn.init.xavier_normal_(w)

    def forward(self, u_hidden, v_hidden, class_values):

        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)

        basis_outputs = []
        for weight in self.weight:
            u_w = torch.matmul(u_hidden, weight)
            x = torch.matmul(u_w, v_hidden.t())
            basis_outputs.append(x)

        outputs = torch.stack(basis_outputs, 2)
        #outputs = torch.matmul(basis_outputs, self.a)
        outputs = outputs + self.u_bias.unsqueeze(1).repeat(1,outputs.size(1),1) \
                          + self.v_bias.unsqueeze(0).repeat(outputs.size(0),1,1)
        outputs = outputs.permute(2,0,1)

        softmax_out = F.softmax(outputs, 0)

        m_hat = torch.stack([(class_values[r])*output for r, output in enumerate(softmax_out)], 0)
        m_hat = torch.sum(m_hat, 0)

        return outputs, m_hat
