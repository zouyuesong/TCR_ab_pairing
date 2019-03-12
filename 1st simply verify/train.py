import time
import os, sys
# import cv2
import numpy as np
import pickle
# import pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
from IPython import embed
import argparse
# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


from utils import toTensor

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', required=True)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--d_pos', type=float, default=20)
parser.add_argument('--d_neg', type=float, default=100)
parser.add_argument('--gap', type=float, default=200)
parser.add_argument('--batch_size', type=float, default=128)
# parser.add_argument('--length', type=int, default=-1)
# parser.add_argument('--count', type=int, default=-1)

opt = parser.parse_args()

# print("name: ", opt.name)

from dataset import create_dataLoader
# TODO: test_loader
train_loader, validate_loader = create_dataLoader(opt)
# validate_data = toTensor(validate_data)

init_lr = 1e-3
from net import create_model
net = create_model(opt.length)
net#.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)

if not os.path.exists(os.path.join('dist_hist', opt.name)):
    os.mkdir(os.path.join('dist_hist', opt.name))
if not os.path.exists(os.path.join('dist_hist', opt.name, 'train')):
    os.mkdir(os.path.join('dist_hist', opt.name, 'train'))
if not os.path.exists(os.path.join('dist_hist', opt.name, 'validate')):
    os.mkdir(os.path.join('dist_hist', opt.name, 'validate'))

# visualize the distribution of dist_pos & dist_neg
def plot_dist(dist_pos, dist_neg, stage, epoch):
    path = os.path.join('dist_hist', opt.name, stage, '%03dtest.jpg'%epoch)
    dist_pos = dist_pos.flatten()
    dist_neg = dist_neg.flatten()

    l = 0
    r = max(max(dist_pos), max(dist_neg))
    plt.subplot(211)
    np, bins, patches = plt.hist(dist_pos,100, label='pos', normed=True)#color=())
    bsp = [(bins[i-1]+bins[i])/2 for i in range(1,len(bins))] 

    nn, bins, patches = plt.hist(dist_neg,100, label='neg', normed=True)#density=1/(opt.batch_size-1))
    bsn = [(bins[i-1]+bins[i])/2 for i in range(1,len(bins))] 

    plt.subplot(212)
    plt.plot(bsp, np)
    plt.plot(bsn, nn)

    # plt.xlabel(Xlabel)
    # plt.ylabel()
    # plt.xlim(l,r)
    plt.savefig(path)
    if stage == 'validate':
        plt.subplot(211)
        plt.cla()
        plt.subplot(212)
        plt.cla()
    

# FIXME: hinge-loss is derived by presume 1, that A-B, C-D, A-D indicating B-C
def HingeLoss(feats_a, feats_p, epoch):
    b = feats_a.size(0)
    eye = (torch.ones(b, b) - torch.eye(b,b))#.cuda()

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
    # actually we can first try power=0

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

    # dist_pos = dist_intra.data.cpu().numpy()
    # dist_neg = dist_inter.data.cpu().numpy()[eye.data.cpu().numpy().astype(bool)]
    dist_pos = dist_intra.data.numpy()
    dist_neg = dist_inter.data.numpy()[eye.numpy().astype(bool)]

    # print("loss: %lf,  dist_pos: %lf, dist_neg: %lf" % (hinge_loss, dist_pos, dist_neg))

    # tb.add_scalar('train/dist_pos_l2', dist_intra.mean(), it)
    # tb.add_scalar('train/dist_neg_l2', (dist_inter*eye).sum()/eye.sum(), it)
    # tb.add_scalar('train/loss_h', hinge_loss, it)
    
    return hinge_loss, dist_pos, dist_neg


# train with hinge-loss
# train with cross entropy
loss_func = nn.CrossEntropyLoss()
print("train start")
for epoch in range(opt.epoch):
    print("epoch =", epoch)
    # TODO: the function for lr adjusting
    # adjust_learning_rate(it)

    optimizer.zero_grad()

    losses, accs, dist_pos, dist_neg = [], [], [], []
    # totally 42 mini-batches
    for step, (pair, label) in enumerate(train_loader):
        a_chain = pair[:,:,:opt.length,:]
        b_chain = pair[:,:,opt.length:,:]
        # a_chain = a_chain.cuda()
        # b_chain = b_chain.cuda()
        # label = label.cuda()

        # HingeLoss stage
        feats_a = net(a_chain)
        feats_b = net(b_chain)
        loss, di_pos, di_neg = HingeLoss(feats_a, feats_b, epoch)

        # # Classification Loss stage
        # output = net(a_chain, b_chain)
        # loss = loss_func(output, label)
        # pred = torch.max(output, 1)[1].data.cpu().numpy().squeeze()
        # acc = sum(pred == label.data.cpu().numpy())/label.shape[0]
        # # print(output[:,0].data.cpu().numpy().mean(), output[:,1].data.cpu().numpy().mean(), label.data.cpu().numpy().mean())
        # # embed()

        losses.append(loss.data.numpy())#.data.cpu().numpy())
        # accs.append(acc)
        # print(loss.data.cpu().numpy(), acc)
        dist_pos.append(di_pos)
        dist_neg.append(di_neg)

        loss.backward()
        optimizer.step()


    # print the process of training 
    losses = np.array(losses)
    accs = np.array(accs)
    print("loss: %lf, acc: %lf" % (losses.mean(), accs.mean()))

    dist_pos = np.array(dist_pos)
    dist_neg = np.array(dist_neg)
    print("pos: %lf, neg: %lf" % (dist_pos.mean(), dist_neg.mean()))
    if epoch % 10 == 0:
        plot_dist(dist_pos, dist_neg, 'train', epoch)

    # validation
    if epoch % 10 == 0:
        net.eval()
        v_dist_pos, v_dist_neg, validate_losses, validate_accs = [], [], [], []
        for step, (pair, label) in enumerate(validate_loader):

            a_chain = pair[:,:,:opt.length,:]
            b_chain = pair[:,:,opt.length:,:]
            # a_chain = a_chain.cuda()
            # b_chain = b_chain.cuda()
            # label = label.cuda()
        
            # HingeLoss stage
            feats_a = net(a_chain)
            feats_b = net(b_chain)
            loss, di_pos, di_neg = HingeLoss(feats_a, feats_b, epoch)
            v_dist_pos.append(di_pos)
            v_dist_neg.append(di_neg)
            # Classification Loss stage
            # output = net(a_chain, b_chain)
            # loss = loss_func(output, label)
            # pred = torch.max(output, 1)[1].data.cpu().numpy().squeeze()
            # acc = sum(pred == label.data.cpu().numpy())/label.shape[0]
            # validate_accs.append(acc)

            validate_losses.append(loss.data.numpy())#.data.cpu().numpy())

        validate_losses = np.array(validate_losses)

        # validate_accs = np.array(validate_accs)
        # print("Validate: loss: %lf, acc: %lf" % (validate_losses.mean(), validate_accs.mean()))
        
        v_dist_pos = np.array(v_dist_pos)
        v_dist_neg = np.array(v_dist_neg)
        print("Test: loss: %lf,  pos: %lf, neg: %lf" % (loss, v_dist_pos.mean(), v_dist_neg.mean()))
        plot_dist(v_dist_pos, v_dist_neg, 'validate', epoch)
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