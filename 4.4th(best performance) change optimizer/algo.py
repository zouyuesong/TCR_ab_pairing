import numpy as np 
from IPython import embed
import torch
from utils import toTensor

def km(n, weight):
    match = []

    return match
    



def match(opt, net, data):
    n = data.shape[0]
    a = toTensor(data[:,:,:opt.length])
    b = toTensor(data[:,:,opt.length:])
    feat_a = net.feature_extractor_x(a)
    feat_b = net.feature_extractor_y(b)
    embed()

    target = [i for i in range(n)]
    weight = [[net.match(feat_a[i], feat_b[j]) for i in range(n)] for j in range(n)]
    # TODO: weight

    # TODO: the method to match a & b
    # 1. greedy
    #   1.1 for each row
    pred = torch.max(weight, 1)[1].data.cpu().numpy().squeeze()
    acc1 = sum(pred == target)/n
    #   1.2 for each column
    pred = torch.max(weight.transpose(0,1), 1)[1].data.cpu().numpy().squeeze()
    acc2 = sum(pred == target)/n

    # 2. km 


    return (acc1, acc2)
    
    
    

