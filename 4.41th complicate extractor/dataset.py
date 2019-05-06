import torch
import torchvision
import numpy as np
import torch.utils.data as Data

from data_preprocess import get_data
from utils import toLongTensor as toTensor

from IPython import embed 

# dataloader
class TDataSet(Data.Dataset):
    def __init__(self, data, label, transform=None):
        self.dataset = data
        self.transform = transform 
        self.label = torch.from_numpy(label).type(torch.float)
    
    def __getitem__(self, idx):
        DNA = self.dataset[idx]
        if self.transform:
            DNA = self.transform(DNA)
        return (DNA, self.label[idx])

    def __len__(self):
        return len(self.dataset)

        
        
def create_dataLoader(opt):
    (train_data, train_label), (validate_data, validate_label) = get_data(opt)

    opt.train_count = len(train_data)
    train_dataset = TDataSet(train_data, train_label, transform=toTensor)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)

    opt.validate_count = len(validate_data)
    validate_dataset = TDataSet(validate_data, validate_label, transform=toTensor)
    validate_loader = Data.DataLoader(dataset=validate_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return train_loader, validate_loader
