import numpy as np
import torch
from IPython import embed

def toTensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor)

def get_BLOSUM50_Matrix():
    d = np.array(['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V'])
    from Bio.SubsMat import MatrixInfo
    blosum = MatrixInfo.blosum50

    blosumMatrix = []
    for i in range(20):
        mm = []
        for j in range(20):
            if (d[i], d[j]) in blosum:
                mm.append(blosum[(d[i], d[j])])
            else:
                mm.append(blosum[(d[j], d[i])])
        blosumMatrix.append(mm)
    return blosumMatrix

blosumMatrix = get_BLOSUM50_Matrix()

def get_onehot(raw_seq):
    length = max([len(seq) for seq in raw_seq])
    d = np.array(['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V'], dtype='|S5')
    y = np.array([np.fromstring(seq.ljust(length, 'N'), dtype='|S1').reshape(-1,1) == d  for seq in raw_seq])
    return y.astype(np.float)

def to_BLOSUM(onehot_seq):
    '''
    DNA (one hot) to BLOSUM matrix
    '''
    y = np.array([np.matmul(DNA, blosumMatrix) for DNA in onehot_seq])
    return y
    
def get_onehot4DNA(raw_seq):
    length = max([len(DNA) for DNA in raw_seq])
    d = np.array(['A','C','G','T','N'], dtype='|S5')
    y = np.array([np.fromstring(DNA.ljust(length, 'N'), dtype='|S1').reshape(-1,1) == d  for DNA in raw_seq])
    return y.astype(np.float)