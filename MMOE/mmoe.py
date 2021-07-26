import numpy as np
import os
from scipy.sparse.construct import rand
import torch
import matplotlib.pyplot as plt

from torch import tensor
from torch._C import get_default_dtype
from torch.types import Number
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import classification_report, roc_curve, auc
from Dataset import MMoEDataset
from early_stopping import EarlyStopping
from mmoe_model import MMOE 

torch.autograd.set_detect_anomaly(True)

def loss_batch(model, loss_fun, xv, y_t1, y_t2, opt=None):
    p_t1, p_t2 = model(xv)
    loss_t1 = loss_fun(p_t1, y_t1)
    loss_t2 = loss_fun(p_t2, y_t2)

    loss = 1*loss_t1 + 1*loss_t2

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), loss_t1.item(), loss_t2.item(), len(xv)

def fit(epochs, model, loss_fun, opt, scheduler, train_dl, valid_dl, valid_ds, early_stopping=None):
    loss_v = []
    loss_t1_v = []
    loss_t2_v = []
    auc_v = []
    y_label = np.array(valid_ds.labels)
    x_valid_data = valid_ds.data
    for epoch in range(epochs):
        model.train()
        for xvb, y_t1b, y_t2b in train_dl:
            loss_batch(model, loss_fun, xvb, y_t1b, y_t2b, opt)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            losses, losses_t1, losses_t2, nums = zip(*[loss_batch(model, loss_fun, xvb, y_t1b, y_t2b) for xvb, y_t1b, y_t2b in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_loss_t1 = np.sum(np.multiply(losses_t1, nums)) / np.sum(nums)
        val_loss_t2 = np.sum(np.multiply(losses_t2, nums)) / np.sum(nums)
        loss_v.append(val_loss)
        loss_t1_v.append(val_loss_t1)
        loss_t2_v.append(val_loss_t2)
        print(epoch, val_loss)
        y_s_prob, _ = model(x_valid_data)
        dnn_fpr, dnn_tpr, thresholds = roc_curve(y_label, y_s_prob.detach().numpy(), pos_label=1)
        dnn_auc = auc(dnn_fpr, dnn_tpr)
        auc_v.append(dnn_auc)

        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    lidx = np.argmin(loss_v)
    lbidx = np.argmin(loss_t1_v)
    lcidx = np.argmin(loss_t2_v)

    print('minimum: ')
    print(f'epoch: {lidx}, loss: {loss_v[lidx]}')
    print(f'epoch: {lbidx}, loss_bce: {loss_t1_v[lbidx]}')
    print(f'epoch: {lcidx}, loss_ce: {loss_t2_v[lcidx]}')
    print(f'epoch: {lidx}, auc: {auc_v[lidx]}')

    ylabels = ['loss', 'auc of expert 1', 'loss of expert 1', 'loss of expert 2']
    indices = [lidx, lidx, lidx, lidx]
    plt_data = [np.array(loss_v), np.array(auc_v), np.array(loss_t1_v), np.array(loss_t2_v)]
    for d, idx, ylabel in zip(plt_data, indices, ylabels):
        plt.figure()
        plt.plot(np.arange(len(d)), d, c='b')
        plt.plot(idx, d[idx], marker='o', c='r')
        plt.xlabel('epoches')
        plt.ylabel(ylabel)
        plt.show()

    # plt.show()
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))              

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = './data/marital-status-train.csv' 
        valid_file = './data/marital-status-valid.csv'
        test_file = './data/marital-status-test.csv'
    else:
        train_file = '.\\data\\marital-status-train.csv'
        valid_file = '.\\data\\marital-status-valid.csv'
        test_file = '.\\data\\marital-status-tests.csv'
    lr = 0.001
    bs = 100
    epochs = 400

    label_cols = ['income', 'marital-status']
    train_ds = MMoEDataset(train_file, label_cols[0], label_cols[1])
    valid_ds = MMoEDataset(valid_file, label_cols[0], label_cols[1], train_ds.scaler)
    test_ds = MMoEDataset(test_file, label_cols[0], label_cols[1], train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)

    cols = train_ds.dataframe.columns.tolist()
  
    model = MMOE(2, 2, 96, 32, 1) # num_experts, num_tasks, num_inputs, num_neurons_expert, num_neurons_tower

    loss_fun = nn.BCELoss()
 
    opt = optim.Adam(model.parameters(), lr=lr,)
    scheduler = ExponentialLR(opt, gamma=0.9)

    patience = 10
    early_stopping = EarlyStopping(patience, verbose=False)

    fit(epochs, model, loss_fun, opt, scheduler, train_dl, valid_dl, valid_ds, early_stopping)
    y_s_prob, y_r_prob = model(test_ds.data)
    y_s_prob = y_s_prob.squeeze().tolist()
    y_s_pred = np.array([0 if y_s_prob[i] < 0.5 else 1 for i in range(len(y_s_prob))])
    y_r_prob = y_r_prob.squeeze().tolist()
    y_r_pred = np.array([0 if y_r_prob[i] < 0.5 else 1 for i in range(len(y_r_prob))])
    # y_r_pred = np.argmax(y_r_prob.tolist(), axis=1)
    y_s_test = test_ds.labels.tolist()
    y_r_test = test_ds.aux_labels.tolist()
    sreport = classification_report(y_s_test, y_s_pred, digits=4)
    print(sreport)
    rreport = classification_report(y_r_test, y_r_pred, digits=4)
    print(rreport)