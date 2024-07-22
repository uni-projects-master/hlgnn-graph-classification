import sys
import os
import torch
from dataReader_utils.dataReader import DGLDatasetReader
from model.LGNet import LGNetwork
from conv.LGCN import LGCN
from conv.GCN import GCN

from impl.graphClassificationImpl import modelImplementation_graphClassificator
from utils.utils_method import printParOnFile
from torch.utils.tensorboard import SummaryWriter
import torch.cuda

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


def run_validation(forceobj=True):
    test_type = 'LGCN'
    # sys setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_list = range(3)
    n_epochs = 300
    test_epoch = 1
    early_stopping_patience = 80

    # test hyper par
    lr_list = [0.01]
    lambda_list = [1e-3]
    weight_decay_list = [0.0]

    # model settings
    dropout_list = [0.0]
    num_layers_list = [4]
    hidden_dim_list = [64]
    activation_list = ['ReLU']
    inc_type_list = ['in']

    # line graph hyper par
    lg_level_list = [1]
    backtrack_list = [False]

    # Set Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset>
    dataset_name = 'MUTAG'
    self_loops = True
    for lg_level in lg_level_list:
        for backtrack in backtrack_list:
            for lr in lr_list:
                for reg_lambda in lambda_list:
                    for num_layers in num_layers_list:
                        for hidden_dim in hidden_dim_list:
                            for activation in activation_list:
                                for dropout in dropout_list:
                                    for inc_type in inc_type_list:
                                        for weight_decay in weight_decay_list:
                                            for run in run_list:
                                                test_name = "run_" + str(run) + '_' + test_type
                                                # Env
                                                test_name = test_name + \
                                                            "_data-" + dataset_name + \
                                                            "_lr-" + str(lr) + \
                                                            "_dropout-" + str(dropout) + \
                                                            "_lambda-" + str(reg_lambda) + \
                                                            "_weight-decay-" + str(weight_decay) + \
                                                            "_num_layer-" + str(num_layers) + \
                                                            "_hidden-dim-" + str(hidden_dim) + \
                                                            "_activation-" + str(activation) + \
                                                            "_lg-level-" + str(lg_level) + \
                                                            "_inc-type-" + str(inc_type) + \
                                                            "_backtrack-" + str(backtrack)

                                                writer = SummaryWriter(
                                                    "./test_log/" + str(test_type) + '/tensorboard/{}'.format(
                                                        test_name))
                                                test_type_folder = os.path.join("./test_log/" + test_type + "/data")
                                                if not os.path.exists(test_type_folder):
                                                    os.makedirs(test_type_folder)
                                                training_log_dir = os.path.join(test_type_folder, test_name)
                                                print(test_name)
                                                if not os.path.exists(training_log_dir):
                                                    os.makedirs(training_log_dir)

                                                    printParOnFile(test_name=test_name, log_dir=training_log_dir,
                                                                   par_list={"dataset_name": dataset_name,
                                                                             "learning_rate": lr,
                                                                             "dropout": dropout,
                                                                             "lambda": reg_lambda,
                                                                             "weight_decay": weight_decay,
                                                                             "num_layers": num_layers,
                                                                             "hidden_dim": hidden_dim,
                                                                             "activation": activation,
                                                                             "lg_level": lg_level,
                                                                             "inc_type": inc_type,
                                                                             "backtrack": backtrack,
                                                                             "test_epoch": test_epoch,
                                                                             "self_loops": self_loops})

                                                    train_loader, val_loader, test_loader, n_feats, n_classes = DGLDatasetReader(
                                                        dataset_name=dataset_name,
                                                        lg_level=lg_level,
                                                        backtrack=backtrack,
                                                        batch_size=64,
                                                        self_loops=self_loops)

                                                    model = LGNetwork(in_feats=n_feats,
                                                                      n_classes=n_classes,
                                                                      dropout=dropout,
                                                                      num_layers=num_layers,
                                                                      h_feats=hidden_dim,
                                                                      activation=activation,
                                                                      inc_type=inc_type,
                                                                      convLayer=GCN,
                                                                      device=device).to(device)

                                                    model_impl = modelImplementation_graphClassificator(model=model,
                                                                                                        criterion=criterion,
                                                                                                        device=device)
                                                    model_impl.set_optimizer(lr=lr, weight_decay=weight_decay)

                                                    model_impl.train_test_model(train_loader,
                                                                                val_loader,
                                                                                test_loader,
                                                                                n_epochs=n_epochs,
                                                                                reg_lambda=reg_lambda,
                                                                                test_epoch=test_epoch,
                                                                                test_name=test_name,
                                                                                log_path=training_log_dir,
                                                                                writer=writer,
                                                                                patience=early_stopping_patience)
                                            else:
                                                print("test has been already execute")


if __name__ == '__main__':
    # print(torch.cuda.device_count())
    # print(torch.cuda.is_available())
    # print(torch.__version__)
    run_validation()
