import time
import os, sys
import numpy as np
import pickle
import pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
from IPython import embed
import argparse
# import cv2
from tqdm import tqdm
from tensorboardX import SummaryWriter

from validating import Validate


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', required=True)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--length', type=int, default=-1)
parser.add_argument('--count', type=int, default=-1)
parser.add_argument('--roc_fig', type=bool, default=True)

opt = parser.parse_args()

# FIXME: for data-flow economy
if 'try' in opt.name:
    opt.roc_fig = False

print("name: ", opt.name)
tb = SummaryWriter(os.path.join('runs/', opt.name))

from dataset import create_dataLoader
# TODO: test_loader
train_loader, validate_loader = create_dataLoader(opt)

from net import create_model
net = create_model(opt.length)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)


# FIXME: previously used CrossEntropyLoss
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.BCELoss()
print("train start")
for epoch in range(opt.epoch):
    print("epoch =", epoch)

    losses, acc = [], 0
    # totally 42 mini-batches
    for step, (pair, label) in enumerate(train_loader):
        # FIXME: half of train data
        # a_chain = torch.from_numpy(np.random.randint(10, size=x[:,:,:opt.length,:].shape)).float()
        # b_chain = torch.from_numpy(np.random.randint(10, size=x[:,:,:opt.length,:].shape)).float()

        a_chain = pair[:,:,:opt.length,:]
        b_chain = pair[:,:,opt.length:,:]
        a_chain = a_chain.cuda()
        b_chain = b_chain.cuda()
        label = label.cuda()

        output = net(a_chain, b_chain)

        # print(pair[:,0,0,0] == label.float())
        
        loss = loss_func(output, label)
        # print(label, output)
        pred = (output.data.cpu().numpy() > 0.5)
        acc += sum(pred == label.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data.cpu().numpy())

    # print the process of training 
    losses = np.array(losses)
    acc /= opt.train_count
    print("loss: %lf, acc: %lf" % (losses.mean(), acc))
    tb.add_scalar('train/train_loss', losses.mean(), epoch)
    tb.add_scalar('train/train_acc', acc, epoch)
    tb.add_scalar('train/train_predict', abs((output-0.5)*2).mean(), epoch)

    # validation
    if epoch % 10 == 0:
        print((output.view(-1)-0.5)*(label-0.5)*4)
        Validate(opt, net, loss_func, validate_loader, tb, epoch)

# final validate
Validate(opt, net, loss_func, validate_loader, tb, epoch)

# TODO: Test phase:

tb.close()
