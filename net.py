import numpy as np
import torch
from torch import nn
import math
from IPython import embed


class 
# FIXME: the simplest basic network
class basicNet(nn.Module):
    """
    input len x type
    """
    def __init__(self, length, type, d=5):
        super(basicNet, self).__init__()
        print(length)
        self.length = length

        # TODO: more sophisicated network
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(7,1), stride=(1,1), padding=(3, 0))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(7,1), stride=(1,1), padding=(3, 0))
        self.init_conv = nn.Conv2d(1, 1, kernel_size=(d, 5), padding=(int((d-1)/2), 0))
        
        # FIXME: Exactly I haven't seen the effect of "bidirectional"
        self.rnn = nn.LSTM(input_size=5, hidden_size=128,num_layers=1, batch_first=True)

        # FIXME: also I'm not really know the mean of channel
        self.fc_final = nn.Linear(length, 512)
        
        self.fc_x = nn.Linear(128, 128)
        self.fc_y = nn.Linear(128, 128)

        # TODO: I'm not sure bi-classificate is Linear(128, 2) or Linear(128, 1)
        self.fc_cls = nn.Linear(512*2, 2)

    def feature_extractor(self, x):

        # # FIXME: first naively try RNN
        # x = x.view(-1, 1, self.length)
        # # embed()
        # x, (h_n,h_c) = self.rnn(x.view(-1, self.length, 5))
        # # embed()
        # x = x[:, -1, :]
        # # embed()

        x = self.init_conv(x)
        x = torch.relu(x)
        x = self.fc_final(x.reshape(-1, self.length))

        return x

    def forward(self, x, y):
        x = self.feature_extractor_x(x)
        y = self.feature_extractor_y(y)

        # x = self.fc_x(x)
        # y = self.fc_y(y)

        # FIXME: the function f(x, y) can vary   *  max  cat 
        # f = torch.max(x, y)
        f = torch.cat((x,y), dim=-1)
        # f = x * y
        # f = torch.relu(f)
        output = self.fc_cls(f)
        # FIXME: try tanh
        # output = nn.Tanh()(output)
        # output = nn.ReLU()(output)
        
        # embed()
        return output

def init_weights(m):
        # print(m)
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, 0.02)

def create_model(length):
    net = basicNet(length, 5)
    net.apply(init_weights)

    return net


if __name__ == "__main__":
    l = 30
    net = basicNet(l)
    net.cuda()
    print(net)
    x = torch.randn(128, 1, l, 5)
    x[:,:,:int(l/2),:] = x[0,:,:int(l/2),:]
    y = torch.randn(128, 1, l, 5)
    x = x.cuda()
    y = y.cuda()
    net(x, y)
    
