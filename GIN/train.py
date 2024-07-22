import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from lineConv import LGNet
from dataReader import DGLDatasetReader
from torch.utils.tensorboard import SummaryWriter


def split_fold10(labels, fold_idx=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx


def evaluate(dataloader, device, model):
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        labels = labels.to(torch.int64)
        labels = labels.to(device)
        feat = []
        for g in batched_graph:
            feat.append(g.ndata["feat"])
            g.to(device)
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


def train(train_loader, val_loader, device, model, writer):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # training loop
    for epoch in range(150):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            labels = labels.to(torch.int64)
            labels = labels.to(device)
            feat = []
            for g in batched_graph:
                feat.append(g.ndata["feat"])
                g.to(device)
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            '''l2_lambda = 0.001
            l2_penalty = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l2_penalty = l2_penalty + torch.norm(param, 2)
            loss += l2_lambda * l2_penalty'''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_loss = total_loss / (batch + 1)
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                epoch, train_loss, train_acc, valid_acc
            )
        )
        writer.add_scalar("train loss", train_loss, epoch)
        writer.add_scalar("train acc", train_acc, epoch)
        writer.add_scalar("val acc", valid_acc, epoch)
        # writer.add_scalar("test acc", test_acc, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="MUTAG",
        choices=["MUTAG", "PTC", "NCI1", "PROTEINS"],
        help="name of dataset (default: MUTAG)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GINConv module with a fixed epsilon = 0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_type = 'GraphClassification'
    train_loader, val_loader, test_loader, in_size, out_size = \
        DGLDatasetReader('MUTAG',
                         lg_level=1,
                         backtrack=False,
                         batch_size=128)

    # model = GIN(in_size, 16, out_size).to(device)
    runs = 1
    for run in range(runs):
        model = LGNet(
            in_feats=in_size,
            h_feats=64,
            out_feats=out_size,
            num_layers=5,
            device=device)

        test_name = "run-" + str(run) + \
                    "_data-" + 'MUTAG' + \
                    "_skips" + \
                    "_numlayers-5" + \
                    "_hidden-dim-64" + \
                    "_lg-level-1" + \
                    "_activation-relu" + \
                    "_norm-batch" + \
                    "_readout-avg" + \
                    "_batch-128" + \
                    "_backtrack-False" + \
                    "_tr-data-50%"

        writer = SummaryWriter("./test_log/" + str(test_type) + '/tensorboard/{}'.format(test_name))

        print("Training...")
        train(train_loader, val_loader, device, model, writer)

    '''# load and split dataset
       dataset = GINDataset(
           args.dataset, self_loop=True, degree_as_nlabel=False
       )  # add self_loop and disable one-hot encoding for input features
       labels = [l for _, l in dataset]
       train_idx, val_idx = split_fold10(labels)

       # create dataloader
       train_loader = GraphDataLoader(
           dataset,
           sampler=SubsetRandomSampler(train_idx),
           batch_size=1,
           pin_memory=torch.cuda.is_available(),
       )
       val_loader = GraphDataLoader(
           dataset,
           sampler=SubsetRandomSampler(val_idx),
           batch_size=1,
           pin_memory=torch.cuda.is_available(),
       )

       # create GIN model
       in_size = dataset.dim_nfeats
       out_size = dataset.gclasses

       '''
