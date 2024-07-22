import os
import sys
import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.conv import GATConv

from torch import nn
from dgl.nn.pytorch.glob import SumPooling

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


def get_inc(g, inc_type):
    return g.inc(inc_type)


class LGCore(nn.Module):
    def __init__(self, in_feats, out_feats, agg_type):
        super(LGCore, self).__init__()
        self.out_feats = out_feats
        self.agg_type = agg_type
        self.convLayer = GraphConv(in_feats, out_feats)
        self.fusionLayer = GraphConv(in_feats, out_feats)
        self.conv_w = nn.Parameter(torch.rand(out_feats))
        self.topDown_w = nn.Parameter(torch.rand(out_feats))
        self.bottomUp_w = nn.Parameter(torch.rand(out_feats))

        self.gat_conv = GATConv(out_feats, out_feats, num_heads=3)
        self.lin = nn.Linear(out_feats*3, out_feats)

        self.batch_norm = nn.BatchNorm1d(out_feats)
        self.layer_norm = nn.LayerNorm(out_feats)
        #self.dropout = nn.Dropout(0.2)

    def forward(self, g, feats, inc):
        g = dgl.add_self_loop(g)
        if self.agg_type == 'top_down':
            curr_h = feats[0]
            next_h = feats[1]
            curr_inc = inc[0]

            conv_layer = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            # conv_layer = self.batch_norm(conv_layer)
            # conv_layer = F.relu(conv_layer)

            top_down_layer = self.fusionLayer(g, torch.mm(curr_inc, next_h)) * self.topDown_w.unsqueeze(0)
            # top_down_layer = self.batch_norm(top_down_layer)
            # top_down_layer = F.relu(top_down_layer)

            result = conv_layer + top_down_layer
            '''print(result.shape)

            result = self.gat_conv(g, result)

            result = torch.cat(result, dim=0)

            result = self.lin(result)'''

        elif self.agg_type == 'bottom_up':
            curr_h = feats[0]
            prev_h = feats[1]
            prev_inc_y = inc[0]

            conv_layer = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            # conv_layer = self.batch_norm(conv_layer)
            # conv_layer = F.relu(conv_layer)

            bottom_up_layer = self.fusionLayer(g, torch.mm(prev_inc_y, prev_h)) * self.bottomUp_w.unsqueeze(0)
            # bottom_up_layer = self.batch_norm(bottom_up_layer)
            # bottom_up_layer = F.relu(bottom_up_layer)

            result = conv_layer + bottom_up_layer

        elif self.agg_type == 'both':
            curr_h = feats[0]
            prev_h = feats[1]
            next_h = feats[2]
            curr_inc = inc[0]
            prev_inc_y = inc[1]

            conv_layer = self.convLayer(g, curr_h) * self.conv_w.unsqueeze(0)
            # conv_layer = self.batch_norm(conv_layer)
            # conv_layer = F.relu(conv_layer)

            bottom_up_layer = self.fusionLayer(g, torch.mm(prev_inc_y, prev_h)) * self.bottomUp_w.unsqueeze(0)
            # bottom_up_layer = self.batch_norm(bottom_up_layer)
            # bottom_up_layer = F.relu(bottom_up_layer)

            top_down_layer = self.fusionLayer(g, torch.mm(curr_inc, next_h)) * self.topDown_w.unsqueeze(0)
            # top_down_layer = self.batch_norm(top_down_layer)
            # top_down_layer = F.relu(top_down_layer)

            result = conv_layer + bottom_up_layer + top_down_layer

        # skip connections and normalisation
        n = self.out_feats // 2
        result = self.layer_norm(result)
        result = torch.cat([result[:, :n], F.relu(result[:, n:])], 1)
        result = self.batch_norm(result)
        #result = F.relu(result)
        #result = self.dropout(result)

        return result


class LGCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LGCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.inc_type = "in"
        self.top_down = LGCore(self.in_feats, self.out_feats, 'top_down')
        self.bottom_up = LGCore(self.in_feats, self.out_feats, 'bottom_up')
        self.both = LGCore(self.in_feats, self.out_feats, 'both')

    def forward(self, lg_list, lg_h):
        new_lg_h = []
        for i in range(0, len(lg_list)):
            g = lg_list[i]
            current_h = lg_h[i]
            # finding the previous graph's hidden embedding
            # if it's the first graph, do only top-down
            if i == 0:
                # get the next graph's embedding
                next_h = lg_h[i + 1]
                # get the incident matrix B_0
                g_inc = get_inc(g, self.inc_type)
                # apply top-down aggregation
                feats = [current_h, next_h]
                inc = [g_inc]
                new_h = self.top_down(g, feats, inc)
                new_lg_h.append(new_h)
            # if it's the last graph in the hierarchy, do only bottom-up
            elif i == (len(lg_list) - 1):
                # get the prev graph's embedding
                prev_h = lg_h[i - 1]
                # get the prev graph's incident matrix
                prev_g = lg_list[i - 1]
                prev_g_inc = get_inc(prev_g, self.inc_type)
                prev_g_inc_y = torch.transpose(prev_g_inc, 0, 1)
                feats = [current_h, prev_h]
                inc = [prev_g_inc_y]
                new_h = self.bottom_up(g, feats, inc)
                new_lg_h.append(new_h)
            # if it's in-between, apply both
            else:
                # get next and prev graphs' embeddings
                prev_h = lg_h[i - 1]
                next_h = lg_h[i + 1]
                feats = [current_h, prev_h, next_h]
                # get curr and prev graphs' inc matrix (B_i, B_(i-1)^T)
                g_inc = get_inc(g, self.inc_type)
                prev_g = lg_list[i - 1]
                prev_g_inc = get_inc(prev_g, self.inc_type)
                prev_g_inc_y = torch.transpose(prev_g_inc, 0, 1)
                inc = [g_inc, prev_g_inc_y]
                # apply aggregation layer
                new_h = self.both(g, feats, inc)
                new_lg_h.append(new_h)

        return new_lg_h


class LGNet(nn.Module):
    def __init__(self,
                 in_feats,
                 h_feats,
                 out_feats,
                 num_layers,
                 device=None,
                 ):
        super(LGNet, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.outLayer = nn.ModuleList()

        for layer in range(num_layers - 1):
            if layer == 0:
                self.layers.append(LGCN(in_feats, h_feats))
            else:
                self.layers.append(LGCN(h_feats, h_feats))

            self.batch_norms.append(nn.BatchNorm1d(h_feats))


        # self.pool = (SumPooling())
        # self.drop = nn.Dropout(0.5)
        self.outLayer = nn.Linear(h_feats, out_feats)

    def forward(self, lg_list, lg_h):
        hidden_rep = [lg_h]
        for layer in self.layers:
            lg_h = layer(lg_list, lg_h)
            # lg_h = self.batch_norms[i](lg_h)
            # lg_h = F.relu(lg_h)
            hidden_rep.append(lg_h)
        # apply readout layer
        embeds = []
        #print(lg_h[0].shape)
        '''for i in range(1, len(hidden_rep)):
            for j in range(len(hidden_rep[i])):
                with lg_list[j].local_scope():
                    lg_list[j].ndata['h'] = hidden_rep[i][j]
                    embeds.append(dgl.mean_nodes(lg_list[j], 'h'))
                # Calculate graph representation by average readout.\
                #embeds.append(dgl.mean_nodes(lg_list[j], 'h'))
        concat_embeds = torch.cat(embeds, dim=1)
        #print(concat_embeds.shape)'''
        h = lg_h[0]
        with lg_list[0].local_scope():
            lg_list[0].ndata['h'] = h
            readout = dgl.mean_nodes(lg_list[0], 'h')
        output = self.outLayer(readout)

        return output
