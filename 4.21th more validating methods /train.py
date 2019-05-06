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
from tensorboardX import SummaryWriter

from utils import toTensor
from validating import Validate


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', required=True)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--d_pos', type=float, default=20)
parser.add_argument('--d_neg', type=float, default=100)
parser.add_argument('--gap', type=float, default=200)
parser.add_argument('--batch_size', type=float, default=256)
parser.add_argument('--length', type=int, default=-1)
parser.add_argument('--count', type=int, default=-1)

opt = parser.parse_args()

print("name: ", opt.name)
tb = SummaryWriter(os.path.join('runs/', opt.name))

from dataset import create_dataLoader
# TODO: test_loader
train_loader, validate_loader = create_dataLoader(opt)
# validate_data = toTensor(validate_data)

init_lr = 5e-3
from net import create_model
net = create_model(opt.length)
# net.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)



def adjust_learning_rate(optimizer, it):
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr


# train with hinge-loss
# train with cross entropy

# FIXME: previously used CrossEntropyLoss
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.BCELoss()
print("train start")
for epoch in range(opt.epoch):
    print("epoch =", epoch)
    # TODO: the function for lr adjusting
    adjust_learning_rate(optimizer, epoch)

    losses, acc = [], 0
    # totally 42 mini-batches
    for step, (pair, label) in enumerate(train_loader):
        # FIXME: half of train data
        # a_chain = torch.from_numpy(np.random.randint(10, size=x[:,:,:opt.length,:].shape)).float()
        # b_chain = torch.from_numpy(np.random.randint(10, size=x[:,:,:opt.length,:].shape)).float()

        a_chain = pair[:,:,:opt.length,:]
        b_chain = pair[:,:,opt.length:,:]
        # a_chain = a_chain.cuda()
        # b_chain = b_chain.cuda()
        # label = label.cuda()

        output = net(a_chain, b_chain)

        # print(pair[:,0,0,0] == label.float())
        
        loss = loss_func(output, label)
        # print(label, output)
        pred = (output.view(-1).data.cpu().numpy() > 0.5)
        acc += sum(pred == label.data.cpu().numpy())
        # loss, di_pos, di_neg = HingeLoss(feat_a, feat_b, epoch)
        # print(output[:,0].data.cpu().numpy().mean(), output[:,1].data.cpu().numpy().mean(), label.data.cpu().numpy().mean())
        # embed()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data.cpu().numpy())

    # print the process of training 
    losses = np.array(losses)
    acc /= opt.train_count
    print("loss: %lf, acc: %lf" % (losses.mean(), acc), "(lr: ", init_lr)
    tb.add_scalar('train/train_loss', losses.mean(), epoch)
    tb.add_scalar('train/train_acc', acc, epoch)
    tb.add_scalar('train/train_predict', abs((output-0.5)*2).mean(), epoch)
    # FIXME: adjust learning rate
    if (losses.mean() < 0.67):
        init_lr = 1e-4
        print("lr change")
    if (losses.mean() < 0.02):
        init_lr = 1e-6
        print("lr 2nd change")


    # validation
    if epoch % 10 == 0:
        print((output.view(-1)-0.5)*(label-0.5)*4)
        Validate(opt, net, loss_func, validate_loader, tb, epoch)

# final validate
Validate(opt, net, loss_func, validate_loader, tb, epoch)

# TODO: Test phase:

tb.close()