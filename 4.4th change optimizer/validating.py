import io
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

from utils import toTensor

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
    net.eval()
    
    n = opt.validate_count
    label = validate_loader.dataset.label
    a_chain = toTensor(validate_loader.dataset.dataset[:,:,:opt.length,:])
    b_chain = toTensor(validate_loader.dataset.dataset[:,:,opt.length:,:])

    output = net(a_chain, b_chain)
    loss = loss_func(output, label).data.numpy()
    output = output.view(-1).data.numpy()
    label = label.data.numpy()

    pred = (output > 0.5)
    acc = (pred == label).mean()
    auc = get_AUC(output, label, n)
    roc = get_ROC(output, label, n)

    print("Validate: loss: %lf, acc: %lf, auc: %lf" % (loss, acc, auc))
    tb.add_scalar('validate/validate_loss', loss, epoch)
    tb.add_scalar('validate/validate_acc', acc, epoch)
    tb.add_scalar('validate/validate_prec', abs((output-0.5)*2).mean(), epoch)
    tb.add_scalar('validate/validate/auc', auc, epoch)
    tb.add_figure('validate/ROC curve', roc, epoch)
    
    net.train()