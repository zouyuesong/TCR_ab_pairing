import numpy as np
from IPython import embed

from utils import toTensor

def get_AUC(outputs, labels, n):
    auc = (outputs[:,np.newaxis] > outputs[np.newaxis,:]) * (labels[:,np.newaxis] > labels[np.newaxis,:])
    auc = auc.mean() * 4
    return auc


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

    print("Validate: loss: %lf, acc: %lf, auc: %lf" % (loss, acc, auc))
    tb.add_scalar('validate/validate_loss', loss, epoch)
    tb.add_scalar('validate/validate_acc', acc, epoch)
    tb.add_scalar('validate/validate_prec', abs((output-0.5)*2).mean(), epoch)
    tb.add_scalar('validate/validate/auc', auc, epoch)
    
    net.train()