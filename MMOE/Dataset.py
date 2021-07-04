import torch
import pandas as pd
# import numpy as np
from torch.utils.data import Dataset#, DataLoader
from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")

class DnnDataset(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, str_label, str_pop=None, scaler=None):
        """
            csv_file (String): data file to be read.
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.scaler = scaler
        if str_pop != None:
            self.dataframe.pop(str_pop)
        data, labels = self.dataframe, self.dataframe.pop(str_label)
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        # self.labels = torch.tensor(labels).float() # lr
        # self.stroke_labels = torch.tensor(stroke_labels).float()
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        # stroke_labels = self.stroke_labels
        labels = self.labels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], labels[idx].unsqueeze(-1)

class MMoEDataset(Dataset):
    """
        selected features of stroke risk study
    """
    # import torch
    def __init__(self, csv_file, str_label, str_aux_label, scaler=None):
        """
            csv_file (String): data file to be read.
            scaler (None or StandardScaler)  : to normalize the data.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.scaler = scaler
        data, labels, aux_labels = self.dataframe, self.dataframe.pop(str_label), self.dataframe.pop(str_aux_label)
        
        if scaler==None:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.scaler = scaler
        else:
            data = scaler.transform(data)
        self.data = torch.tensor(data).float()
        self.labels = torch.tensor(labels).float()
        self.aux_labels = torch.tensor(aux_labels).float()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        # stroke_labels = self.stroke_labels
        labels = self.labels
        aux_labels = self.aux_labels
        # return data[idx], lables[idx].unsqueeze(-1) # lr
        return data[idx], labels[idx].unsqueeze(-1), aux_labels[idx].unsqueeze(-1)
        # return data[idx], labels[idx], aux_labels[idx]