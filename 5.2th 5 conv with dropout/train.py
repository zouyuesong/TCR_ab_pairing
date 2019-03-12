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
# from tensorboardX import SummaryWriter

from utils import toTensor

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--name', required=True)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--d_pos', type=float, default=20)
parser.add_argument('--d_neg', type=float, default=100)
parser.add_argument('--gap', type=float, default=200)
parser.add_argument('--batch_size', type=float, default=256)
parser.add_argument('--length', type=int, default=-1)
parser.add_argument('--count', type=int, default=-1)

opt = parser.parse_args()

# print("name: ", opt.name)

from dataset import create_dataLoader
# TODO: test_loader
train_loader, validate_loader = create_dataLoader(opt)
# validate_data = toTensor(validate_data)

init_lr = 1e-5
from net import create_model
net = create_model(opt.length)
# net.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)


# FIXME: hinge-loss is derived by presume 1, that A-B, C-D, A-D indicating B-C
def HingeLoss(feats_a, feats_p, epoch):
    b = feats_a.size(0)
    eye = (torch.ones(b, b) - torch.eye(b,b))# .cuda()

    # FIXME: be careful here
    a = feats_a.float()
    p = feats_p.float()
    a2 = (a**2).sum(dim=1).reshape((-1, 1))
    p2 = (p**2).sum(dim=1).reshape((1, -1))
    ap = torch.matmul(a, p.transpose(0,1))

    dist_intra = ((a-p)**2).sum(dim=1).reshape((-1,1))
    dist_inter = (a2 + p2 - 2 * ap)

    gap = 50
    power = epoch / opt.epoch * 18
    # power = 0
    # FIXME: actually we can first try power=0

    d = dist_intra-dist_inter + gap
    J_IvsQ = (d * (d > 0).float()) * eye / 100.
    J_IvsQ_hardness = J_IvsQ + (J_IvsQ < 1e-7).float()*(-1e3)
    J_IvsQ_probs = torch.nn.Softmax(dim=1)(J_IvsQ_hardness*power)
    hinge_dist_IvsQ = (J_IvsQ*J_IvsQ_probs.detach()).sum(dim=1)

    d = dist_intra-dist_inter.transpose(1, 0) + gap
    J_QvsI = (d * (d > 0).float()) * eye / 100.
    J_QvsI_hardness = J_QvsI + (J_QvsI < 1e-7).float()*(-1e3)
    J_QvsI_probs = torch.nn.Softmax(dim=1)(J_QvsI_hardness*power)
    hinge_dist_QvsI = (J_QvsI*J_QvsI_probs.detach()).sum(dim=1)

    hinge_loss = ((hinge_dist_IvsQ+hinge_dist_QvsI)*0.5).mean().float()

    dist_pos = dist_intra.mean()
    dist_neg = (dist_inter*eye).sum()/eye.sum()

    # print("loss: %lf,  dist_pos: %lf, dist_neg: %lf" % (hinge_loss, dist_pos, dist_neg))

    # tb.add_scalar('train/dist_pos_l2', dist_intra.mean(), it)
    # tb.add_scalar('train/dist_neg_l2', (dist_inter*eye).sum()/eye.sum(), it)
    # tb.add_scalar('train/loss_h', hinge_loss, it)
    
    return hinge_loss, dist_pos, dist_neg

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

    optimizer.zero_grad()

    losses, accs, dist_pos, dist_neg = [], [], [], []
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
        acc = sum(pred == label.data.cpu().numpy())/label.shape[0]
        # loss, di_pos, di_neg = HingeLoss(feat_a, feat_b, epoch)
        # print(output[:,0].data.cpu().numpy().mean(), output[:,1].data.cpu().numpy().mean(), label.data.cpu().numpy().mean())
        # embed()

        loss.backward()
        optimizer.step()

        losses.append(loss.data.cpu().numpy())
        accs.append(acc)
        # print(loss.data.cpu().numpy(), acc)
        # dist_pos.append(di_pos.data.cpu().numpy())
        # dist_neg.append(di_neg.data.cpu().numpy())

    # print the process of training 
    losses = np.array(losses)
    accs = np.array(accs)
    print("loss: %lf, acc: %lf" % (losses.mean(), accs.mean()), "(lr: ", init_lr)
    # FIXME: adjust learning rate
    if (losses.mean() < 0.67):
        init_lr = 1e-4
        print("lr change")
    if (losses.mean() < 0.02):
        init_lr = 1e-6
        print("lr 2nd change")

    # dist_pos = np.array(dist_pos)
    # dist_neg = np.array(dist_neg)
    # print("pos: %lf, neg: %lf" % (dist_pos.mean(), dist_neg.mean()))

    # validation
    if epoch % 10 == 0:
        print((output.view(-1)-0.5)*(label-0.5)*4)
        net.eval()
        validate_losses, validate_accs = [], []
        for step, (pair, label) in enumerate(validate_loader):

            a_chain = pair[:,:,:opt.length,:]
            b_chain = pair[:,:,opt.length:,:]
            # a_chain = a_chain.cuda()
            # b_chain = b_chain.cuda()
            # label = label.cuda()
        
            output = net(a_chain, b_chain)
            loss = loss_func(output, label)
            pred = torch.max(output, 1)[1].data.cpu().numpy().squeeze()
            acc = sum(pred == label.data.cpu().numpy())/label.shape[0]
            validate_losses.append(loss.data.cpu().numpy())
            validate_accs.append(acc)

        validate_losses = np.array(validate_losses)
        validate_accs = np.array(validate_accs)
        print("Validate: loss: %lf, acc: %lf" % (validate_losses.mean(), validate_accs.mean()))
        
        # loss, dist_pos, dist_neg = HingeLoss(feat_a, feat_b, epoch)
        # print("Test: loss: %lf,  pos: %lf, neg: %lf" % (loss, dist_pos, dist_neg))
        # loss, dist_pos, dist_neg = HingeLoss(feat_a, feat_b, 0)
        # print("Test with power 0: loss: %lf,  pos: %lf, neg: %lf" % (loss, dist_pos, dist_neg))
        net.train()

# final validate
validate_losses, validate_accs = [], []
for step, (pair, label) in enumerate(validate_loader):
    # print(step)

    a_chain = pair[:,:,:opt.length,:]
    b_chain = pair[:,:,opt.length:,:]
    # a_chain = a_chain.cuda()
    # b_chain = b_chain.cuda()
    # label = label.cuda()

    output = net(a_chain, b_chain)
    loss = loss_func(output, label)
    pred = torch.max(output, 1)[1].data.cpu().numpy().squeeze()
    acc = sum(pred == label.data.cpu().numpy())/label.shape[0]
    validate_losses.append(loss.data.cpu().numpy())
    validate_accs.append(acc)

validate_losses = np.array(validate_losses)
validate_accs = np.array(validate_accs)
print("Validate: loss: %lf, acc: %lf" % (validate_losses.mean(), validate_accs.mean()))

# loss, dist_pos, dist_neg = HingeLoss(feat_a, feat_b, epoch)
# print("Test: loss: %lf,  pos: %lf, neg: %lf" % (loss, dist_pos, dist_neg))
# loss, dist_pos, dist_neg = HingeLoss(feat_a, feat_b, 0)
# print("Test with power 0: loss: %lf,  pos: %lf, neg: %lf" % (loss, dist_pos, dist_neg))