import os
import torch
import numpy as np
from torch.nn import Module
import time
from utils.utils_method import prepare_log_files
import torch.nn.functional as F
import warnings
import sys
from torch.utils.tensorboard import SummaryWriter
from numba import jit, cuda

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

warnings.filterwarnings("ignore", category=UserWarning)


class modelImplementation_graphClassificator(Module):
    def __init__(self, model, criterion, device=None):
        super(modelImplementation_graphClassificator, self).__init__()
        self.optimizer = None
        self.model = model
        self.criterion = criterion
        self.device = device

    def set_optimizer(self, lr, weight_decay=0):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_test_model(self, train_loader, val_loader, test_loader, reg_lambda, n_epochs,
                         test_epoch,
                         writer, test_name="", log_path=".", patience=30):

        train_log, test_log, valid_log = prepare_log_files(test_name, log_path)
        dur = []
        best_val_acc = 0.0
        best_val_loss = 100000.0
        no_improv = 0

        for epoch in range(n_epochs):
            if no_improv > patience:
                break
            self.model.train()
            epoch_start_time = time.time()
            total_loss = 0.0
            total_acc = 0.0
            num_batches = len(train_loader)
            for batched_graphs, batched_labels in train_loader:
                batched_graphs = batched_graphs.to(self.device)
                batched_labels = batched_labels.long().to(self.device)
                model_out = self.model(batched_graphs, batched_graphs.ndata["attr"].float())
                loss = self.criterion(model_out, batched_labels)
                #total_acc += (model_out.argmax(1) == batched_labels).sum().item()
                '''l2_lambda = reg_lambda
                l2_penalty = torch.tensor(0., requires_grad=True)
                for param in self.model.parameters():
                    l2_penalty = l2_penalty + torch.norm(param, 2)
                loss += l2_lambda * l2_penalty'''
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / num_batches

            cur_epoch_time = time.time() - epoch_start_time
            dur.append(cur_epoch_time)

            train_acc = self.evaluate(train_loader)
            val_acc = self.evaluate(val_loader)
            test_acc = self.evaluate(test_loader)

            if epoch % 5 == 0:
                print("epoch : ", epoch, " -- loss: ", train_loss, "-- time: ", cur_epoch_time)
                print("training acc : ", train_acc, " -- test_acc : ", test_acc, " -- valid_acc : ", val_acc)
                #print("training loss : ", train_loss, " -- test_loss : ", test_loss, " -- valid_loss : ", val_loss)
                print("------")
                mean_epoch_time = np.mean(np.asarray(dur))
                train_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        train_loss,
                        train_acc,
                        mean_epoch_time))
                train_log.flush()
                test_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        test_acc,
                        mean_epoch_time))
                test_log.flush()
                valid_log.write(
                    "{:d}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        val_acc,
                        mean_epoch_time))
                valid_log.flush()
                writer.add_scalar("train loss", train_loss, epoch)
                writer.add_scalar("train acc", train_acc, epoch)
                writer.add_scalar("val acc", val_acc, epoch)
                writer.add_scalar("test acc", test_acc, epoch)
            # early stopping
            no_improv += 1
            if val_acc > best_val_acc:
                no_improv = 0
                #best_val_loss = val_loss
                best_val_acc = val_acc
                # print("--ES--")
                # print("save_new_best_model, with acc:", val_acc)
                # print("------")
                self.save_model(test_name, log_path)

        # print("Best val acc:", best_val_acc)
        # print("Best val loss:", best_val_loss)
        self.load_model(test_name, log_path)
        # print("-----BEST EPOCH RESULT-----")
        # _, train_acc = self.evaluate(input_features, labels, train_mask)
        # _, val_acc = self.evaluate(input_features, labels, valid_mask)
        # _, test_acc = self.evaluate(input_features, labels, test_mask)
        # print("training acc : ", train_acc, " -- test_acc : ", test_acc, " -- valid_acc : ", val_acc)

    def save_model(self, test_name, log_folder='./'):
        torch.save(self.model.state_dict(), os.path.join(log_folder, test_name + '.pt'))

    def load_model(self, test_name, log_folder):
        self.model.load_state_dict(
            torch.load(os.path.join(log_folder, test_name + '.pt'), map_location=torch.device('cpu')))

    '''def evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():
            correct = 0.0
            loss = 0.0
            num_graphs = len(loader.dataset)
            for batch_graphs, batch_labels in loader:
                batch_graphs = batch_graphs.to(self.device)
                batch_labels = batch_labels.long().to(self.device)
                out = self.model(batch_graphs, batch_graphs.ndata["attr"].float())
                pred = out.argmax(dim=1)
                loss += self.criterion(out, batch_labels).item()
                correct += (pred == batch_labels).sum().item()
            return loss / num_graphs, correct / num_graphs'''

    def evaluate(self, dataloader):
        self.model.eval()
        total = 0
        total_correct = 0
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(self.device)
            labels = labels.to(self.device)
            feat = batched_graph.ndata.pop("attr")
            total += len(labels)
            logits = self.model(batched_graph, feat)
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
        acc = 1.0 * total_correct / total
        return acc
