from torch.utils.data import Dataset
import random
import h5py
import torch
import numpy as np
from scipy.io import loadmat

from torchvision import transforms
transformations = transforms.Compose([transforms.ToTensor()]) #It also scales to 0-1 by dividing by 255.

def preProcessInhouseMtx(folderPath):

    d5 = loadmat(folderPath)
    n1, n2 = 32, 16

    d5Gt = d5['myMtx'].reshape(-1, n1, n2)
    d5Crp = d5Gt[:, 4:30, 3:]

    n1C, n2C = d5Crp.shape[1], d5Crp.shape[2]

    selFreq = np.linalg.norm(d5Crp.reshape(-1, n1C * n2C), axis = 1) / (2 * n1C * n2C)**(1/2)
    selElems = selFreq > 5
    d5CrpSel = d5Crp[selElems, :, :]

    return d5CrpSel, selElems, d5['rNew']

def loadMtxExp(folderPath):
    n1 = 26
    n2 = 13
    selFreqBeg = 1
    selFreqStep = 3
    expMtx = loadmat(folderPath)['Aconcat']
    newSize = [-1, n1, n2]
    expMtx = expMtx.reshape(newSize)
    expMtx = torch.Tensor(expMtx).float().reshape(newSize).cuda()
    return expMtx
    
class MRAdatasetH5NoScale(Dataset): #Scaling: allPatchesOfAllSubjects =/ max(allPatchesOfAllSubjects )
    def __init__(self, filePath, transform = transformations, prefetch = True, dim = 2, device=None ):
        super(Dataset, self).__init__()
        self.h5f = h5py.File(filePath, 'r')
        self.keys = list(self.h5f.keys())
        
        self.prefetch = prefetch
        if device is None:
            device = torch.device('cuda')
        if (self.prefetch):
            self.data = torch.zeros((len(self.keys), 1, *(np.array(self.h5f[self.keys[0]])).shape[-dim:]))    
            for ii in range(len(self.keys)):
                self.data[ii] = torch.tensor(np.array(self.h5f[self.keys[ii]]))
            self.data = self.data.to(device).float() / self.data.float().max()
            self.h5f.close()
        else:
            self.transform = transform
            random.shuffle(self.keys)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        theIndex = index % len(self.keys)
        if self.prefetch:
            return self.data[theIndex]
        else:    
            key = self.keys[theIndex]
            data = np.array(self.h5f[key])
            if self.transform:
                data = self.transform(data)
            return data
    def openFile(self, filePath):
        self.h5f = h5py.File(filePath, 'r')
    def closeFile(self):
        self.h5f.close()