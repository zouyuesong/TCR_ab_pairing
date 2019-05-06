import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython import embed
from tqdm import tqdm

from utils import toLongTensor as toTensor

def get_AUC(outputs, labels, n):
    auc = (outputs[:,np.newaxis] > outputs[np.newaxis,:]) * (labels[:,np.newaxis] > labels[np.newaxis,:])
    auc = auc.mean() * 4
    return auc


def gen_plot(FPR, TPR):
    ''' Create a pyplot plot and save to buffer. '''
    plt.figure()
    plt.plot(FPR, TPR)
    plt.title('ROC curve')
    return plt.gcf()
    
def get_ROC(outputs, labels, n):
    n = int(n/2)
    t = np.vstack((outputs, labels))
    t = t.T[np.lexsort(-t[::-1,:])].T
    FPR = (np.where(t[1])[0] - np.arange(n))/n
    TPR = (np.arange(n) + 1)/n
    roc = gen_plot(FPR, TPR)
    return roc
    


def Validate(opt, net, loss_func, validate_loader, tb, epoch):
    print("validate(epoch = %d): " % epoch, end='')
    net.eval()
    
    with torch.no_grad():
        n = opt.validate_count
        outputs = torch.zeros(0)
        for step, (pair, label) in enumerate(validate_loader):
            a_chain = pair[:,:,:opt.length,:]
            b_chain = pair[:,:,opt.length:,:]
            a_chain = a_chain.cuda()
            b_chain = b_chain.cuda()

            output = net(a_chain, b_chain)
            outputs = torch.cat((outputs, output.detach().cpu()))

        labels = validate_loader.dataset.label
        loss = loss_func(outputs, labels).data.numpy()
        outputs = outputs.data.numpy()
        labels = labels.data.numpy()

        preds = (outputs > 0.5)
        acc = (preds == labels).mean()
        auc = get_AUC(outputs, labels, n)
        roc = get_ROC(outputs, labels, n)

        print("Validate : loss: %lf, acc: %lf, auc: %lf" % (loss, acc, auc))
        tb.add_scalar('validate/validate_loss', loss, epoch)
        tb.add_scalar('validate/validate_acc', acc, epoch)
        tb.add_scalar('validate/validate_prec', abs((output-0.5)*2).mean(), epoch)
        tb.add_scalar('validate/validate/auc', auc, epoch)
        if opt.roc_fig:
            tb.add_figure('validate/ROC curve', roc, epoch)
    
    net.train()
