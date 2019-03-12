import numpy as np
import torch
from torch import nn
import math
from IPython import embed

class feature_extractor(nn.Module):
    """
    input : DNA_length x type
    output: feature_dim 
    """
    def __init__(self, length, type, out_dim):
        super(feature_extractor, self).__init__()
        self.length = length
        d = 5
        self.init_conv = nn.Conv2d(1, 1, kernel_size=(d, type), padding=(int((d-1)/2), 0))
        self.init_bn = nn.BatchNorm2d(1)


        # TODO: more sophisticated network required
        # FIXME: use conv layers as sequence scale-contractor, then go through a LSTM 

        self.conv1 = nn.Conv1d(1, 4, kernel_size=5, stride=3, padding=2)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=5, stride=3, padding=2)
        self.bn2 = nn.BatchNorm1d(16)
        # self.conv3 = nn.Conv1d(8, 16, kernel_size=5, stride=3, padding=2)
        # self.bn3 = nn.BatchNorm1d(16)

        # self.final_l = 67 #16*int(length/8)
        self.rnn = nn.LSTM(input_size=16, hidden_size=128,num_layers=1, batch_first=True)
        self.out = nn.Linear(128, out_dim)
        # self.fc_final = nn.Linear(self.final_l, out_dim)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = torch.relu(x)
        x = x.view(x.size(0), 1, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = torch.relu(x)

        x = torch.nn.AvgPool1d(2)(x)

        x, (h_n, h_c) = self.rnn(x.transpose(-2,-1))
        result = self.out(x[:, -1, :])
        return result
        
# FIXME: the simplest basic network
class basicNet(nn.Module):
    """
    input len x type
    """
    def __init__(self, length, type, d=5):
        super(basicNet, self).__init__()
        print(length)
        self.length = length
        
        # FIXME: different feat_extractor for x & y,   or the same ? 
        self.feat_length = 128
        self.feature_extractor_x = feature_extractor(length, type, self.feat_length)
        self.feature_extractor_y = feature_extractor(length, type, self.feat_length)

        self.fc_x = nn.Linear(self.feat_length, self.feat_length)
        self.fc_y = nn.Linear(self.feat_length, self.feat_length)

        self.fc_cls = nn.Linear(self.feat_length, 1)


    def forward(self, x, y):
        x = self.feature_extractor_x(x)
        y = self.feature_extractor_y(y)

        # x = self.fc_x(x)
        # y = self.fc_y(y)

        # FIXME: the function f(x, y) can vary   *  max  cat 
        # f = torch.max(x, y)
        # f = torch.cat((x,y), dim=-1)
        f = x * y
        # f = torch.relu(f)
        output = self.fc_cls(f)
        output = torch.sigmoid(output)

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
    
