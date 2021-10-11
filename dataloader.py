import os
import numpy as np

from torch.utils.data import Dataset

class DenoiseDataset(Dataset):

    def __init__(self, fold_100, fold_25, fold_5, datatype='SD'):

        self.fold_100 = []
        self.fold_25 = []
        self.fold_5 = []
        self.dtype = datatype

        for file in os.listdir(fold_100):
            self.fold_100.append(os.path.join(fold_100, file))
            self.fold_25.append(os.path.join(fold_25, file))
            self.fold_5.append(os.path.join(fold_5, file))

    def __len__(self):
        return len(self.fold_100)

    def __getitem__(self, idx):

        it_100 = np.expand_dims(np.load(self.fold_100[idx]), axis=0)
        it_25 = np.expand_dims(np.load(self.fold_25[idx]), axis=0)
        it_5 = np.expand_dims(np.load(self.fold_5[idx]), axis=0)

        if self.dtype=='SD':
            return it_100.astype(np.float32), it_25.astype(np.float32)
        elif self.dtype=='LD':
            return it_25.astype(np.float32), it_5.astype(np.float32)

class DenoiseDataset3D(Dataset):

    def __init__(self, fold_100, fold_25, fold_5, thickness=19, datatype='SD'):

        self.fold_100 = []
        self.fold_25 = []
        self.fold_5 = []
        self.dtype = datatype
        self.thickness = thickness

        for file in os.listdir(fold_100):
            self.fold_100.append(os.path.join(fold_100, file))
            self.fold_25.append(os.path.join(fold_25, file))
            self.fold_5.append(os.path.join(fold_5, file))

    def __len__(self):
        return len(self.fold_100)

    def __getitem__(self, idx):

        it_100 = np.expand_dims(np.load(self.fold_100[idx]), axis=0)
        it_25 = np.expand_dims(np.load(self.fold_25[idx]), axis=0)
        it_5 = np.expand_dims(np.load(self.fold_5[idx]), axis=0)

        if self.thickness < it_100.shape[1]:
            cont = int((it_100.shape[1] - self.thickness)/2)
            it_100 = it_100[:, cont:-cont]
            it_25 = it_25[:, cont:-cont]
            it_5 = it_5[:, cont:-cont]

        if self.dtype=='SD':
            return it_25.astype(np.float32), it_100.astype(np.float32)
        elif self.dtype=='LD':
            return it_5.astype(np.float32), it_25.astype(np.float32)