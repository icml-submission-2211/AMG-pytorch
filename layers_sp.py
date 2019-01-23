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

        self.n_hop = n_hop
        self.alpha = nn.Parameter(torch.tensor(0., requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(0., requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(0., requires_grad=True))

        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.u_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim))
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

    def sparse_eye(self,size):
        i = torch.LongTensor([[i for i in range(size)] for j in range(2)])
        v = torch.FloatTensor([1 for i in range(size)])
        return torch.sparse.FloatTensor(i,v,(size,size))

    def normalize(self, adjacencies):
        #adjacencies : sparse list
        #adj_norm : sparse list

        adj_tot = adjacencies[0]
        for i in range(1, len(adjacencies)):
            adj_tot += adjacencies[i]

        eye = self.sparse_eye(adj_tot.size(0)).requires_grad_(False).to(adj_tot.device)
        degree_u = torch.sparse.sum(adj_tot+eye,[1])
        degree_value = 1/torch.sqrt(degree_u._values()-1)
        degree_value[torch.isinf(degree_value)] = 0.
        degree = torch.sparse.FloatTensor(eye._indices(),degree_value ,eye.size()).to(adj_tot.device)
        adj_norm = [torch.sparse.mm(degree,torch.sparse.mm(adj,degree.to_dense())).to_sparse() for adj in adjacencies]

        return adj_norm

    def sparse_transpose(self,sp_A):
        # make transpose of sparse matrix A (rank 3)
        sp_A_T_indices = torch.cat([sp_A.coalesce().indices()[0].view(1,-1),sp_A.coalesce().indices()[2].view(1,-1),sp_A.coalesce().indices()[1].view(1,-1) ],dim=0)
        sp_A_T_size = torch.Size([sp_A.size()[0],sp_A.size()[2],sp_A.size()[1]])
        sp_A_T = torch.sparse.FloatTensor(sp_A_T_indices,sp_A.coalesce().values(),sp_A_T_size)
        return sp_A_T

    def n_sum(self, A):
        # A : dense

        A = [A[i,:,:].to_sparse() for i in range(A.shape[0])]

        A_T = [A[i].transpose(1,0) for i in range(len(A))]
        sp_mx_size = torch.Size([sum(A[0].size()),sum(A[0].size())])
        sp_mx_index_x = [torch.cat([A[i]._indices()[0:1,:], A[i]._indices()[1:,:]+A[0].size()[0]], dim=0)
                         for i in range(len(A))]
        sp_mx_index_y = [torch.cat([A_T[i]._indices()[0:1,:]+A[0].size()[0], A_T[i]._indices()[1:,:]], dim=0)
                         for i in range(len(A))]
        sp_mx_indices = [torch.cat([sp_mx_index_x[i], sp_mx_index_y[i]],dim=1) for i in range(len(A))]
        sp_mx_values = [torch.cat([A[i]._values(), A[i]._values()], dim=0) for i in range(len(A))]
        sp_mx = [torch.sparse.FloatTensor(sp_mx_indices[i],sp_mx_values[i], sp_mx_size) for i in range(len(A))]

        sp_square = [torch.sparse.mm(sp_mx[i],sp_mx[i].to_dense()).to_sparse() for i in range(len(A))]

        sp_cube = [torch.sparse.mm(sp_mx[i],sp_mx[i].to_dense()).to_sparse() for i in range(len(A))]
        sp_cube = [torch.sparse.mm(sp_cube[i],sp_mx[i].to_dense()).to_sparse() for i in range(len(A))]
        sp_cube = [sp_cube[i]+(sp_cube[i].t_()) for i in range(len(A))]

        sp_quad = [torch.sparse.mm(sp_mx[i],sp_mx[i].to_dense()).to_sparse() for i in range(len(A))]
        sp_quad = [torch.sparse.mm(sp_quad[i],sp_mx[i].to_dense()).to_sparse() for i in range(len(A))]
        sp_quad = [torch.sparse.mm(sp_quad[i],sp_mx[i].to_dense()).to_sparse() for i in range(len(A))]

        single = self.normalize(sp_mx)
        square = self.normalize(sp_square)
        cube = self.normalize(sp_cube)
        quad = self.normalize(sp_quad)

        alpha = (1.+torch.tanh(self.alpha))*0.5
        beta = (1.+torch.tanh(self.beta))*0.5*(1.-alpha)
        gamma = (1.+torch.tanh(self.gamma))*0.5*(1.-alpha-beta)
        delta = 1.-alpha-beta#-gamma

        support_norm = [torch.sparse.FloatTensor(single[i]._indices(), alpha*single[i]._values(), single[i].size()) +
                        torch.sparse.FloatTensor(square[i]._indices(), beta*square[i]._values(), square[i].size())
                        torch.sparse.FloatTensor(cube[i]._indices(), gamma*cube[i]._values(), cube[i].size())
                        torch.sparse.FloatTensor(quad[i]._indices(), delta*quad[i]._values(), quad[i].size())
                        for i in range(len(single))]

        return support_norm

    def forward(self, u_feat, v_feat, support):
        # support : dense
        num_rating = support.size(0)
        u_feat = self.dropout(u_feat)
        v_feat = self.dropout(v_feat)
        feat = torch.cat((u_feat, v_feat), 0)
        weight = 0
        support_norm = self.n_sum(support)
        del support
        #support_norm : sparse list
        supports = []

        for r in range(num_rating):
            weight = weight + self.u_weight[r]

            support = torch.sparse.mm(support_norm[r],feat)
            supports.append(torch.matmul(support,weight))

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
