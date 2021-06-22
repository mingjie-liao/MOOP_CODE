import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt

from torch import tensor
from torch.types import Number
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
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

    def forward(self, xv):
        ## experts
        out_experts = torch.zeros(self.num_neurons_expert, self.num_experts) # TODO: MLiao, determine the shape
        for i in range(self.num_experts):
            out_experts[:,i] = getattr(self, 'expert'+str(i))(xv)
        ## gates and weighted opinions
        gates = torch.zeros(self.num_tasks, self.num_experts)
        input_towers = torch.zeros(self.num_tasks, self.num_neurons_expert)
        for i in range(self.num_tasks):
            gates[i, :] = getattr(self, 'gate'+str(i))(xv)
            input_towers[i, :] = torch.mul(gates[i,:].unsqueeze(dim=1), out_experts).sum() # TODO: MLiao, check the multiplication
        ## towers
        out_towers = torch.zeros(self.num_tasks, self.num_neurons_tower)
        for i in range(self.num_tasks):
            out_towers[i, :] = getattr(self, 'tower'+str(i))(input_towers[i,:])
        return torch.sigmoid(out_towers) # TODO: MLiao, check the shapes, ensure element-wise operation

class MoeMC(nn.Module):
    def __init__(self, mc_dims, bc_dims, drop_rate=0.2):
        super().__init__()
        self.mc_dims = mc_dims
        self.bc_dims = bc_dims

        ## Multi-classification part
        self.mc = nn.Sequential(
            nn.Linear(mc_dims[0], mc_dims[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate)
        )
        self.mcOut = nn.Linear(mc_dims[-2], mc_dims[-1])
        ## Binary-classification part
        self.bc = nn.Sequential(
            nn.Linear(bc_dims[0], bc_dims[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate)
        )
        # self.bcOut = nn.Linear(bc_dims[-2], bc_dims[-1])
        ## Gates
        self.GATE1lin = nn.Linear(mc_dims[0],2)
        self.GATE1softmax = nn.Softmax(dim=1) 

    def forward(self, xv):
        # Multi-classification part
        xmc = self.mc(xv)
        # Binary-classification part
        xbc = self.bc(xv)
        # gate
        # x = torch.stack([torch.rand(xmc.shape), torch.rand(xbc.shape)])
        x = torch.rand(xmc.shape)
        g = self.GATE1lin(xv)
        g = self.GATE1softmax(g)
        x = g[:,0].unsqueeze(dim=1)*xmc + g[:,1].unsqueeze(dim=1)*xbc
        x = self.mcOut(x)
        return torch.softmax(x)

class MoeBC(nn.Module):
    def __init__(self, mc_dims, bc_dims, drop_rate=0.2):
        super().__init__()
        self.mc_dims = mc_dims
        self.bc_dims = bc_dims

        ## Multi-classification part
        self.mc = nn.Sequential(
            nn.Linear(mc_dims[0], mc_dims[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate)
        )
        # self.mcOut = nn.Linear(mc_dims[-2], mc_dims[-1])
        ## Binary-classification part
        self.bc = nn.Sequential(
            nn.Linear(bc_dims[0], bc_dims[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate)
        )
        self.bcOut = nn.Linear(bc_dims[-2], bc_dims[-1])
        ## Gates
        self.GATE2lin = nn.Linear(bc_dims[0],2)
        self.GATE2softmax = nn.Softmax(dim=1) 

    def forward(self, xv):
        # Multi-classification part
        xmc = self.mc(xv)
        # Binary-classification part
        xbc = self.bc(xv)
        # gate
        # x = torch.stack([torch.rand(xmc.shape), torch.rand(xbc.shape)])
        x = torch.rand(xbc.shape)
        g = self.GATE2lin(xv)
        g = self.GATE2softmax(g)
        x = g[:,0].unsqueeze(dim=1)*xmc + g[:,1].unsqueeze(dim=1)*xbc
        return torch.sigmoid(self.bcOut(x))

def loss_batch(model, loss_mc_fun, loss_bc_fun, xv, ys, yr, opt=None):
    p_bc, p_mc = model(xv)
    loss_bc = loss_bc_fun(p_bc, ys)
    loss_mc = loss_mc_fun(p_mc, yr)

    loss = 1*loss_bc + 1*loss_mc

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), loss_bc.item(), loss_mc.item(), len(xv)#, lhs.sum()/len(x) # TODO: check loss format and values

def fit(epochs, model, loss_mc_fun, loss_bc_fun, opt, train_dl, valid_dl, valid_ds, early_stopping=None):
    loss_v = []
    loss_bc_v = []
    loss_mc_v = []
    auc_v = []
    y_slabel = np.array(valid_ds.slabels)
    x_valid_data = valid_ds.data
    for epoch in range(epochs):
        model.train()
        for xvb, ysb, yrb in train_dl:
            loss_batch(model, loss_mc_fun, loss_bc_fun, xvb, ysb, yrb, opt)

        model.eval()
        with torch.no_grad():
            losses, losses_bc, losses_mc, nums = zip(*[loss_batch(model, loss_mc_fun, loss_bc_fun, xvb, ysb, yrb) for xvb, ysb, yrb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_loss_bce = np.sum(np.multiply(losses_bc, nums)) / np.sum(nums)
        val_loss_ce = np.sum(np.multiply(losses_mc, nums)) / np.sum(nums)
        loss_v.append(val_loss)
        loss_bc_v.append(val_loss_bce)
        loss_mc_v.append(val_loss_ce)
        # print(epoch, val_loss)
        y_s_prob, _ = model(x_valid_data)
        dnn_fpr, dnn_tpr, thresholds = roc_curve(y_slabel, y_s_prob.detach().numpy(), pos_label=1)
        dnn_auc = auc(dnn_fpr, dnn_tpr)
        auc_v.append(dnn_auc)

        if early_stopping is not None:  
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    lidx = np.argmin(loss_v)
    lbidx = np.argmin(loss_bc_v)
    lcidx = np.argmin(loss_mc_v)
    # aucidx = np.argmin(auc_v)
    print('minimum: ')
    print(f'epoch: {lidx}, loss: {loss_v[lidx]}')
    print(f'epoch: {lbidx}, loss_bce: {loss_bc_v[lbidx]}')
    print(f'epoch: {lcidx}, loss_ce: {loss_mc_v[lcidx]}')
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

    plt.show()
    if (early_stopping is not None) and (early_stopping.early_stop):
        model.load_state_dict(torch.load('checkpoint.pt'))              

if __name__ == '__main__':
    if os.name == 'posix':
        train_file = './data/train_en.csv'  # MLiao: 20 features only! 
        valid_file = './data/valid_en.csv'
        test_file = './data/test_en.csv'
    else:
        train_file = '.\\data\\train_en.csv'
        valid_file = '.\\data\\valid_en.csv'
        test_file = '.\\data\\test_en.csv'
    lr = 0.01
    bs = 100
    epochs = 400
    train_ds = MoEDataset(train_file)
    valid_ds = MoEDataset(valid_file, train_ds.scaler)
    test_ds = MoEDataset(test_file, train_ds.scaler)

    train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=bs)
    test_dl = DataLoader(dataset=test_ds, batch_size=bs)

    cols = train_ds.dataframe.columns.tolist()

    mc_dims = [20, 11, 4]  # MLiao: compatible with dataset loaded
    bc_dims = [20, 11, 1]
  
    model = MOE(mc_dims, bc_dims)

    loss_bc_fun = nn.BCELoss()
    loss_mc_fun = nn.CrossEntropyLoss()
 
    opt = optim.Adam(model.parameters(), lr=lr,)

    patience = 20
    early_stopping = EarlyStopping(patience, verbose=False)

    fit(epochs, model, loss_mc_fun, loss_bc_fun, opt, train_dl, valid_dl, valid_ds, early_stopping)
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

    paraDict = model.state_dict()

    mmc = MoeMC(mc_dims, bc_dims)
    mmcDict = mmc.state_dict()
    pretrained_dict = {k: v for k, v in paraDict.items() if k in mmcDict}
    mmcDict.update(pretrained_dict)
    mmc.load_state_dict(mmcDict)

    mbc = MoeBC(mc_dims, bc_dims)
    mbcDict = mbc.state_dict()
    pretrained_dict = {k: v for k, v in paraDict.items() if k in mbcDict}
    mbcDict.update(pretrained_dict)
    mbc.load_state_dict(mbcDict)