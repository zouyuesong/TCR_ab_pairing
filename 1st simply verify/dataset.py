import torch
import torchvision
import numpy as np
import torch.utils.data as Data

from data_preprocess import data_preprocess
from utils import toTensor

from IPython import embed 

# dataloader
class TDataSet(Data.Dataset):
    def __init__(self, data, label, transform=None):
        self.dataset = data
        self.transform = transform 
        self.label = torch.from_numpy(label).type(torch.long)
    
    def __getitem__(self, idx):
        DNA = self.dataset[idx]
        if self.transform:
            DNA = self.transform(DNA)
        return (DNA, self.label[idx])

    def __len__(self):
        return len(self.dataset)

# pair (a, +) --> tuple (a, +, -)
def sample_negative(data):
    c = len(data)
    l = int(data.shape[2]/2)
    negs = set()
    while len(negs) < c:
        a = np.random.randint(len(data))
        n = np.random.randint(len(data))
        while a == n:
            a = np.random.randint(len(data))
            n = np.random.randint(len(data))
        negs.add((a,n))

    data_neg = np.array([np.concatenate((data[a,:,:l,:], data[n,:,l:,:]), axis=1) for (a,n) in negs])
    return np.concatenate((data, data_neg))
    
def preprocess(data, all_posi=True):
    if all_posi: 
        c = len(data)
        label = np.ones(c)
        return (data, label)
        
    data_neg = sample_negative(data)
    label = np.concatenate((np.zeros(c), np.ones(c)))

    data = np.concatenate((data, data_neg))
    
    ids = np.array([i for i in range(2*c)])
    np.random.shuffle(ids)

    label_shuffled = np.array([label[i] for i in ids])
    data_shuffled = np.array([data[i] for i in ids])
    
    # embed()
    return (data_shuffled, label_shuffled)
        
        
def create_dataLoader(opt):
    train, validate, opt.length = data_preprocess()

    train_data, train_label = preprocess(train)
    validate_data, validate_label = preprocess(validate)

    opt.train_count = len(train_data)
    train_dataset = TDataSet(train_data, train_label, transform=toTensor)
    # FIXME: shuffle = False for debugging
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    opt.validate_count = len(validate_data)
    validate_dataset = TDataSet(validate_data, validate_label, transform=toTensor)
    validate_loader = Data.DataLoader(dataset=validate_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    return train_loader, validate_loader