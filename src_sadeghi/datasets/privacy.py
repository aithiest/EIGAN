import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


class Adult(Dataset):

    def __init__(self, root, train=True,
                 transform=None):
        self.data = sio.loadmat(root + 'data.mat')
        if train:
            X = torch.from_numpy(self.data['X']).float()
            Y = torch.from_numpy(self.data['Y'])
            S = torch.from_numpy(self.data['S'])

            self.X = torch.t(X)
            self.Y = torch.t(Y)
            self.S = torch.t(S)
            # import pdb
            # pdb.set_trace()

        else:
            X = torch.from_numpy(self.data['X_test']).float()
            Y = torch.from_numpy(self.data['Y_test'])
            S = torch.from_numpy(self.data['S_test'])

            self.X = torch.t(X)
            self.Y = torch.t(Y)
            self.S = torch.t(S)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X, Y, S = self.X[index], self.Y[index].long(), self.S[index].long()
        return X, [Y, S]
