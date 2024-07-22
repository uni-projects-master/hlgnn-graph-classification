import os
import sys
import dgl
import numpy as np
import torch
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
os.environ["DGLBACKEND"] = "pytorch"

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


def DGLDatasetReader(dataset_name, lg_level, backtrack, batch_size, self_loops=False, device=None):
    dataset = dgl.data.GINDataset(dataset_name, self_loop=True)
    print("Node feature dimensionality:", dataset.dim_nfeats)
    print("Number of graph categories:", dataset.gclasses)
    #print(len(dataset))

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - num_val - num_training

    indices = list(range(len(dataset)))
    train_indices = indices[:num_training]
    val_indices = indices[num_training:num_training + num_val]
    test_indices = indices[num_training + num_val:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False
    )
    val_loader = GraphDataLoader(
        dataset, sampler=val_sampler, batch_size=batch_size, drop_last=False
    )
    test_loader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False
    )

    #for batched_graph, labels in train_loader:
        #print(batched_graph, labels)

    return train_loader, val_loader, test_loader, dataset.dim_nfeats, dataset.gclasses


def create_line_graphs(g, in_feats, lg_level, backtrack, device):
    # g.edata['_ID'] = {}
    line_graphs = [g]
    # number of nodes in every generated line graph
    # num_node_list = []
    # num_node_list.append(g.num_nodes())
    # print(g.ndata['_ID'])
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


#DGLDatasetReader('MUTAG', lg_level=1, backtrack=False, batch_size=64)
