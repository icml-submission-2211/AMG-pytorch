import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torchvision import models
from layers_sp import *
from metrics import expected_rmse, softmax_accuracy, softmax_cross_entropy

class AMG(nn.Module):
    def __init__(self, num_users, num_items, num_side_features, nb,
                       u_features, v_features, u_features_side, v_features_side,
                       input_dim, class_values, emb_dim, hidden, dropout, **kwargs):
        super(AMG, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_classes = len(class_values)

        self.u_features = u_features
        self.v_features = v_features
        self.u_features_side = u_features_side
        self.v_features_side = v_features_side

        self.class_values = class_values
        self.dropout = dropout

        self.denseu1 = nn.Linear(num_side_features+input_dim, emb_dim, bias=True)
        self.densev1 = nn.Linear(num_side_features+input_dim, emb_dim, bias=True)
        self.gcl1 = GraphConvolution(emb_dim, hidden[0],
                                     self.num_classes, 2, torch.tanh, self.dropout, bias=True)
        self.denseu2 = nn.Linear(hidden[0], hidden[1], bias=True)
        self.densev2 = nn.Linear(hidden[0], hidden[1], bias=True)

        self.bilin_dec = BilinearMixture(num_users=num_users, num_items=num_items,
                                         num_classes=self.num_classes,
                                         input_dim=hidden[1],
                                         nb=nb, dropout=0.)

        for w in [self.denseu1, self.denseu2, self.densev1, self.densev2]:
            nn.init.xavier_normal_(w.weight)


    def forward(self, r_matrix):

        u_f = torch.cat((self.u_features, self.u_features_side), 1)
        v_f = torch.cat((self.v_features, self.v_features_side), 1)
        u_z = torch.relu(self.denseu1(u_f))
        v_z = torch.relu(self.densev1(v_f))

        u_z, v_z = self.gcl1(u_z, v_z, r_matrix)

        u_h = self.denseu2(F.dropout(u_z, self.dropout))
        v_h = self.densev2(F.dropout(v_z, self.dropout))

        output, m_hat = self.bilin_dec(u_h, v_h, self.class_values)

        loss = softmax_cross_entropy(output, r_matrix.float())
        rmse_loss = expected_rmse(m_hat, r_matrix.long(), self.class_values)

        return output, loss, rmse_loss
