import os
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv

os.environ["DGLBACKEND"] = "pytorch"


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.conv = GraphConv(in_feats, out_feats)
        #self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, features):
        h = self.conv(g, features)
        h = F.relu(h)
        return h
