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

        self.fcn_1 = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(p=0.5),
                    nn.Linear(512, 2),
                    )

    def forward(self, x_pos, x_ang):

        # data normalization
        N, C, T, V, M = x_pos.size()
        x_pos = x_pos.permute(0, 4, 3, 1, 2).contiguous()
        x_pos = x_pos.view(N * M, V * C, T)
        x_pos = self.gcn_pos.data_bn(x_pos)
        x_pos = x_pos.view(N, M, V, C, T)
        x_pos = x_pos.permute(0, 1, 3, 4, 2).contiguous()
        x_pos = x_pos.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.gcn_pos.st_gcn_networks, self.gcn_pos.edge_importance):
            x_pos, _ = gcn(x_pos, self.gcn_pos.A * importance)

        # global pooling
        x_pos = F.avg_pool2d(x_pos, x_pos.size()[2:])
        x_pos = x_pos.view(N, M, -1, 1, 1).mean(dim=1)
        #print("x_pos.size(): ", x_pos.size())
        # prediction
        #x_pos_pre = self.gcn_pos.fcn(x_pos)
        x_pos = x_pos.view(x_pos.size(0), -1)
        #x_pos = self.fcn_1(x_pos)
        #x_pos_pre = self.fcn_2(x_pos)
        #x_pos_pre = x_pos_pre.view(x_pos_pre.size(0), -1)

        # data normalization
        N, C, T, V, M = x_ang.size()
        x_ang = x_ang.permute(0, 4, 3, 1, 2).contiguous()
        x_ang = x_ang.view(N * M, V * C, T)
        x_ang = self.gcn_ang.data_bn(x_ang)
        x_ang = x_ang.view(N, M, V, C, T)
        x_ang = x_ang.permute(0, 1, 3, 4, 2).contiguous()
        x_ang = x_ang.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.gcn_ang.st_gcn_networks, self.gcn_ang.edge_importance):
            x_ang, _ = gcn(x_ang, self.gcn_ang.A * importance)

        # global pooling
        x_ang = F.avg_pool2d(x_ang, x_ang.size()[2:])
        x_ang = x_ang.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        #x_ang_pre = self.gcn_ang.fcn(x_ang)
        x_ang = x_ang.view(x_ang.size(0), -1)
        #x_ang = self.fcn_1(x_ang)
        #x_ang_pre = self.fcn_2(x_ang)

        x_pos = torch.cat((x_pos,x_ang),1)
        predict = self.fcn_1(x_pos)

        return predict
