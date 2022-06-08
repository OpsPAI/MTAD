import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader

def load_SMD(data, label, batch_size):

    #mean_df = data.mean(axis=0)
    #std_df = data.std(axis=0)

    #data = pd.DataFrame((data-mean_df)/std_df)
    n_sensor = data.shape[1]
    #data = data.dropna(axis=1)
    data = np.array(data)

    train_df = data[:int(0.5*len(data))]
    train_label = label[:int(0.5*len(data))]

    val_df = data[int(0.5*len(data)):int(0.7*len(data))]
    val_label = label[int(0.5*len(data)):int(0.7*len(data))]
    
    test_df = data[int(0.7*len(data)):]
    test_label = label[int(0.7*len(data)):]

    train_loader = DataLoader(SMD(train_df, train_label), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(SMD(val_df,val_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SMD(test_df,test_label), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor


class SMD(Dataset):
    def __init__(self, data, label, window_size=60, stride_size=1):
        super(SMD, self).__init__()
        self.data = data
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(data, label)

    def preprocess(self, data, label):
        start_idx = np.arange(0, len(data) - self.window_size, self.stride_size)
        end_idx = np.arange(self.window_size, len(data), self.stride_size)

        return data, start_idx, label[end_idx]

    def __len__(self):
        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size, -1, 1])

        return torch.FloatTensor(data).transpose(0, 1)
