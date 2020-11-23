import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn_ui_prmd import Model as ST_GCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.gcn_pos = ST_GCN(*args, **kwargs)
        self.gcn_ang = ST_GCN(*args, **kwargs)

    def forward(self, x_pos, x_ang):

        predict, feature = self.gcn_pos.extract_feature(x_pos)

        # data normalization
        N, C, T, V, M = x_ang.size()
        x_ang = x_ang.permute(0, 4, 3, 1, 2).contiguous()
        x_ang = x_ang.view(N * M, V * C, T)
        x_ang = self.gcn_ang.data_bn(x_ang)
        x_ang = x_ang.view(N, M, V, C, T)
        x_ang = x_ang.permute(0, 1, 3, 4, 2bruce).contiguous()
        x_ang = x_ang.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.gcn_ang.st_gcn_networks, self.gcn_ang.edge_importance):
            x_ang, _ = gcn(x_ang, self.gcn_ang.A * importance)

        _, c, t, v = x_ang.size()
        x_ang = x_ang.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        x_ang = x_ang * feature #+ x_ang

        x_ang = x_ang.view(N, c, v, t, M).permute(0, 4, 1, 2, 3)
        x_ang = x_ang.view(N * M, c, v, t)

        # global pooling
        x_ang = F.avg_pool2d(x_ang, x_ang.size()[2:])
        x_ang = x_ang.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x_ang_pre = self.gcn_ang.fcn(x_ang)
        x_ang_pre = x_ang_pre.view(x_ang_pre.size(0), -1)


        return x_ang_pre
