from datetime import datetime
import time 
import argparse
import torch

from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--pooling_ratio', type=float, default=0.6, help='pooling_ratio')
parser.add_argument('--conv_channel1', type=int, default=16, help='conv_channel1')
parser.add_argument('--conv_channel2', type=int, default=16, help='conv_channel2')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='oneil.pkl')

args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
kge_dim = args.kge_dim
pooling_ratio = args.pooling_ratio
conv_channel1 = args.conv_channel1
conv_channel2 = args.conv_channel2

lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
weight_decay = args.weight_decay
data_size_ratio = args.data_size_ratio
pkl_name = args.pkl_name
device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)


def save_metrics(metrics, filename):
    with open(filename, 'a') as f:
        f.write(','.join(map(str, metrics)) + '\n')

def do_compute(batch, device, model):

    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)  # PR_AUC
    ap = metrics.average_precision_score(target, probas_pred)
    bacc = (recall + (1 - precision)) / 2
    prec = sum(target) / len(target)
    tnr = sum((target == 0) & (pred == 0)) / sum(target == 0)
    kappa = metrics.cohen_kappa_score(target, pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap, bacc, prec, tnr, kappa


def train(model, train_data_loader, loss_fn, optimizer, n_epochs, device, scheduler=None):
    max_acc = 0
    print('Starting training at', datetime.today())
    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        train_probas_pred = []
        train_ground_truth = []

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision, train_recall, train_int_ap, train_ap, train_bacc, train_prec, train_tnr, train_kappa = do_compute_metrics(
                train_probas_pred, train_ground_truth)

            if train_acc > max_acc:
                max_acc = train_acc
                torch.save(model, pkl_name)

        if scheduler:
            scheduler.step()

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}'
              f' train_acc: {train_acc:.4f}')
        # print(f'\t\ttrain_roc: {train_auc_roc:.4f}, train_precision: {train_precision:.4f}')
        # print(f'\t\ttrain_bacc: {train_bacc:.4f}, train_prec: {train_prec:.4f}')
        # print(f'\t\ttrain_tnr: {train_tnr:.4f}, train_kappa: {train_kappa:.4f}')

def test(test_data_loader, model):
    test_probas_pred = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap, test_bacc, test_prec, test_tnr, test_kappa = do_compute_metrics(
            test_probas_pred, test_ground_truth)

    test_metrics = [test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap, test_bacc, test_prec, test_tnr, test_kappa]

    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')
    print(f'\t\ttest_bacc: {test_bacc:.4f}, test_prec: {test_prec:.4f}')
    print(f'\t\ttest_tnr: {test_tnr:.4f}, test_kappa: {test_kappa:.4f}')

    return test_metrics

file_result = './results/folds.txt'
all_metrics = ('fold\ttest_acc\ttest_auc_roc\ttest_f1\ttest_precision\ttest_recall\ttest_int_ap\ttest_ap\ttest_bacc\ttest_prec\ttest_tnr\ttest_kappa')
with open(file_result, 'w') as f:
    f.write(all_metrics + '\n')
for i in range(5):
    train_file = 'data/folds/folds' + str(i) + '/train.csv'
    test_file = 'data/folds/folds' + str(i) + '/test.csv'
    print(f'数据集文件为{train_file, test_file}')

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    train_tup = [(da, db, cell, label) for da, db, cell, label in zip(df_train['drug_a_name'], df_train['drug_b_name'], df_train['cell_line'], df_train['label'])]
    test_tup = [(da, db, cell, label) for da, db, cell, label in zip(df_test['drug_a_name'], df_test['drug_b_name'], df_test['cell_line'], df_test['label'])]

    train_data = DrugDataset(train_tup, ratio=data_size_ratio)
    test_data = DrugDataset(test_tup)

    print(f"Training with {len(train_data)} samples, and testing with {len(test_data)}")

    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_loader = DrugDataLoader(test_data, batch_size=batch_size * 3, num_workers=2)

    model = models.DVRL(n_atom_feats, n_atom_hid, kge_dim, pooling_ratio, conv_channel1, conv_channel2,
                        heads_out_feat_params=[64, 64, 64], blocks_params=[2, 2, 2])
    loss = custom_loss.SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    model.to(device=device)
    train(model, train_data_loader, loss, optimizer, n_epochs, device, scheduler)
    test_model = torch.load(pkl_name)
    test_metrics = test(test_data_loader, test_model)
    fold_num = 'fold' + str(i)
    test_metrics.insert(0, fold_num)
    save_metrics(test_metrics, file_result)



