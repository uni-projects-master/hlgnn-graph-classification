import dgl
from torch import nn
import torch


# This is done for every graph, not the batch
class LGNetwork(nn.Module):
    def __init__(self,
                 in_feats,
                 h_feats,
                 n_classes,
                 inc_type,
                 num_layers,
                 activation,
                 dropout,
                 convLayer,
                 out_fun=nn.Softmax(dim=1),
                 device=None,
                 norm=None,
                 bias=False):
        super(LGNetwork, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        self.in_feats = in_feats
        self.out_fun = out_fun
        self.num_layer = num_layers

        self.layers.append(convLayer(in_feats, h_feats))
        for i in range(num_layers):
            self.layers.append(convLayer(h_feats, h_feats))

        self.outLayer = nn.Linear(h_feats, n_classes)

    def reset_lg_feats(self, feat_0):
        feat_list = []
        for i in range(len(self.lg_node_list)):
            if i == 0:
                feat_list.append(feat_0)
            else:
                feat_list.append(torch.zeros(self.lg_node_list[i], self.in_feats).to(self.device))
        return feat_list

    def forward(self, g, features):
        lg_h = features
        for layer in self.layers:
            lg_h = layer(g, lg_h)
        # apply readout layer
        with g.local_scope():
            g.ndata['h'] = lg_h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')

        return self.outLayer(hg)
