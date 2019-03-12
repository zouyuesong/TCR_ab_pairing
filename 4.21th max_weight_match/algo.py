import numpy as np 
from IPython import embed
import torch
from net import basicNet

def km(n, weight):
    match = []

    return match
    



def predict(net, n, a, b):
    feat_a = net.feature_extractor(a)
    feat_b = net.feature_extractor(b)

    weight = [[net.match(feat_a[i,i+1], feat_b[j,j+1]) for i in range(n)] for j in range(n)]
    embed()
    # TODO: weight

    # TODO: the method to match a & b
    # 1. greedy

    # 2. km 

    match = []


    return match
    
    
    

