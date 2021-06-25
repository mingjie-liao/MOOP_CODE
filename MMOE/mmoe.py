import numpy as np
import os
from scipy.sparse.construct import rand
# import sys
import torch
import matplotlib.pyplot as plt

from torch import tensor
from torch.types import Number
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import classification_report, roc_curve, auc
from Dataset import MMoEDataset
from early_stopping import EarlyStopping

class MMOE(nn.Module):
    def __init__(self, num_experts, num_tasks, num_inputs, num_neurons_expert, num_neurons_tower, drop_rate=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_inputs = num_inputs
        self.num_neurons_expert = num_neurons_expert
        self.num_neurons_tower = num_neurons_tower

        ## Experts
        for i in range(num_experts):
            setattr(self, 'expert'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_neurons_expert),
                nn.ReLU(),
                nn.Dropout(p=drop_rate)
            ))
        ## Gates
        for i in range(num_tasks): # number of towers, fixed to 2.
            setattr(self, 'gate'+str(i), nn.Sequential(
                nn.Linear(num_inputs, num_experts),
                nn.Softmax(dim=1)
            ))
        ## Towers
        for i in range(num_tasks):
            setattr(self, 'tower'+str(i), nn.Sequential(
                nn.Linear(num_neurons_expert, num_neurons_tower),
                nn.ReLU(),
                nn.Dropout(p=drop_rate)
            ))

    def forward(self, xv, bs=10):
        ## experts
        out_experts = torch.zeros(bs, self.num_experts, self.num_neurons_expert)
        for i in range(self.num_experts):
            out_experts[:,i,:] = getattr(self, 'expert'+str(i))(xv)
        ## gates and weighted opinions
        gates = torch.zeros(bs, self.num_experts, self.num_tasks)
        input_towers = torch.zeros(bs, self.num_neurons_expert, self.num_tasks)
        for i in range(self.num_tasks):
            gates[:,:,i] = getattr(self, 'gate'+str(i))(xv)
            for j in range(bs):
                input_towers[j,:,i] = torch.mul(gates[j,:,i].unsqueeze(dim=1), out_experts[j,:,:]).sum(dim=0) 
        ## towers
        out_towers = torch.zeros(bs, self.num_neurons_tower, self.num_tasks)
        for i in range(self.num_tasks):
            out_towers[:,:,i] = getattr(self, 'tower'+str(i))(input_towers[:,:,i])
        out_towers.squeeze_(dim=1) # MLiao: for this case when num_neurons_tower = 1..
        # return torch.sigmoid(out_towers) # TODO: MLiao, check the shapes, ensure element-wise operation
        # return torch.sigmoid(out_towers[:,0]), torch.sigmoid(out_towers[:,1])
        output = torch.sigmoid(out_towers)
        return output[:,0], output[:,1]

def loss_batch(model, loss_fun, xv, y_t1, y_t2, opt=None):
    p_t1, p_t2 = model(xv)
    loss_t1 = loss_fun(p_t1, y_t1)
    loss_t2 = loss_fun(p_t2, y_t2)

    loss = 1*loss_t1 + 1*loss_t2

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), loss_t1.item(), loss_t2.item(), len(xv)#, lhs.sum()/len(x) # TODO: check loss format and values

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
        # print(epoch, val_loss)
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
    # aucidx = np.argmin(auc_v)
    print('minimum: ')
    print(f'epoch: {lidx}, loss: {loss_v[lidx]}')
    print(f'epoch: {lbidx}, loss_bce: {loss_t1_v[lbidx]}')
    print(f'epoch: {lcidx}, loss_ce: {loss_t2_v[lcidx]}')
    print(f'epoch: {lidx}, auc: {auc_v[lidx]}')

    # ylabels = ['loss', 'auc of expert 1', 'loss of expert 1', 'loss of expert 2']
    # indices = [lidx, lidx, lidx, lidx]
    # plt_data = [np.array(loss_v), np.array(auc_v), np.array(loss_bc_v), np.array(loss_mc_v)]
    # for d, idx, ylabel in zip(plt_data, indices, ylabels):
    #     plt.figure()
    #     plt.plot(np.arange(len(d)), d, c='b')
    #     plt.plot(idx, d[idx], marker='o', c='r')
    #     plt.xlabel('epoches')
    #     plt.ylabel(ylabel)
    #     plt.show()

    # plt.show()
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))              

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = './data/marital-status-train.csv'  # MLiao: 20 features only! 
        valid_file = './data/marital-status-valid.csv'
        test_file = './data/marital-status-test.csv'
    else:
        train_file = '.\\data\\marital-status-train.csv'
        valid_file = '.\\data\\marital-status-valid.csv'
        test_file = '.\\data\\marital-status-tests.csv'
    lr = 0.001
    bs = 10
    epochs = 400

    label_cols = ['income', 'marital-status']
    train_ds = MMoEDataset(train_file, label_cols[0], label_cols[1])
    valid_ds = MMoEDataset(valid_file, label_cols[0], label_cols[1], train_ds.scaler)
    test_ds = MMoEDataset(test_file, label_cols[0], label_cols[1], train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)

    cols = train_ds.dataframe.columns.tolist()
  
    model = MMOE(8, 2, 96, 32, 1) # num_experts, num_tasks, num_inputs, num_neurons_expert, num_neurons_tower

    loss_fun = nn.BCELoss()
 
    opt = optim.Adam(model.parameters(), lr=lr,)
    scheduler = ExponentialLR(opt, gamma=0.9)

    patience = 20
    early_stopping = EarlyStopping(patience, verbose=False)

    fit(epochs, model, loss_fun, opt, scheduler, train_dl, valid_dl, valid_ds, early_stopping)
    y_s_prob, y_r_prob = model(test_ds.data)
    y_s_prob = y_s_prob.squeeze().tolist()
    y_s_pred = np.array([0 if y_s_prob[i] < 0.5 else 1 for i in range(len(y_s_prob))])
    y_r_pred = np.argmax(y_r_prob.tolist(), axis=1)
    y_r_test = test_ds.rlabels.tolist()
    y_s_test = test_ds.slabels.tolist()
    sreport = classification_report(y_s_test, y_s_pred, digits=4)
    print(sreport)
    rreport = classification_report(y_r_test, y_r_pred, digits=4)
    print(rreport)

    # paraDict = model.state_dict()

    # mmc = MoeMC(mc_dims, bc_dims)
    # mmcDict = mmc.state_dict()
    # pretrained_dict = {k: v for k, v in paraDict.items() if k in mmcDict}
    # mmcDict.update(pretrained_dict)
    # mmc.load_state_dict(mmcDict)

    # mbc = MoeBC(mc_dims, bc_dims)
    # mbcDict = mbc.state_dict()
    # pretrained_dict = {k: v for k, v in paraDict.items() if k in mbcDict}
    # mbcDict.update(pretrained_dict)
    # mbc.load_state_dict(mbcDict)