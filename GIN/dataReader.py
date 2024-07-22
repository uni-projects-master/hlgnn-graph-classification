import os
import sys
import dgl
import numpy as np
import torch
from dgl.data import LegacyTUDataset
from dgl.data import GINDataset

from dgl.dataloading import GraphDataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

os.environ["DGLBACKEND"] = "pytorch"

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


def DGLDatasetReader(dataset_name, lg_level, backtrack, batch_size, self_loops=False, device=None):
    dataset = dgl.data.LegacyTUDataset(dataset_name)
    num_classes = dataset.num_classes
    g, label = dataset[0]
    feat_dim = g.ndata['feat'].shape[1]
    # Creating line graphs
    for i in range(len(dataset.graph_lists)):
        line_graphs = create_line_graphs(dataset.graph_lists[i], feat_dim, lg_level, backtrack, device)
        dataset.graph_lists[i] = line_graphs

    num_training = int(len(dataset) * 0.5)
    num_val = int(len(dataset) * 0.4)
    num_test = len(dataset) - num_val - num_training

    indices = list(range(len(dataset)))
    train_indices = indices[:num_training]
    val_indices = indices[num_training:num_training + num_val]
    test_indices = indices[num_training + num_val:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
    val_loader = GraphDataLoader(dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False)
    test_loader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)

    '''for lg_list, label in train_loader:
        print("label: ", label)
        #print(lg_list.ndata.pop("feat"))
        for lg in range(len(lg_list)):
            print("lg number ", lg)
            print(lg_list[lg].ndata["feat"])
            #print(lg_list[lg])
            #print(lg_list)'''

    return train_loader, val_loader, test_loader, feat_dim, dataset.num_classes


def create_line_graphs(g, in_feats, lg_level, backtrack, device):
    line_graphs = [g]
    current_g = g

    for t in range(0, lg_level):
        lg = current_g.line_graph(backtracking=backtrack).to(device)
        lg.ndata['feat'] = torch.zeros(lg.num_nodes(), in_feats)
        lg.ndata['_ID'] = lg.nodes()
        lg.edata['_ID'] = torch.tensor(np.array([i for i in range(lg.num_edges())])).to(torch.int64)
        line_graphs.append(lg)
        # num_node_list.append(lg.num_nodes())
        current_g = lg

    return line_graphs


def load_data(dataset_name, self_loop=True):
    if dataset_name == 'PROTEINS':
        return GINDataset('PROTEINS', self_loop)
    elif dataset_name == 'COLLAB':
        return GINDataset('COLLAB', self_loop)
    elif dataset_name == 'MUTAG':
        return GINDataset('MUTAG', self_loop)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))


DGLDatasetReader('MUTAG', lg_level=1, backtrack=False, batch_size=64)
