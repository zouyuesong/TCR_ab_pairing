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
        input_size = 5
        hidden_size = 123
        hidden_size_2 = int(out_dim/2)

        self.embedding = nn.Embedding(input_size, hidden_size)
        # Translate
        # self.init_conv = nn.Conv1d(1, 1, kernel_size=3, stride=3)
        self.init_conv = nn.Conv2d(1, 1, kernel_size=(3,5), stride=(3,3))
        self.rnn1 = nn.GRU(40, hidden_size_2, bidirectional=True)
        
        self.conv1 = nn.Conv1d(1, 4, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(4)

        '''
        self.conv2 = nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2) #150
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=5, stride=3, padding=2) #50
        self.bn3 = nn.BatchNorm1d(16)

        self.conv4 = nn.Conv1d(16, 64, kernel_size=5, stride=3, padding=2) #17
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=5, stride=3, padding=2) #5
        self.bn5 = nn.BatchNorm1d(128)

        self.conv6 = nn.Conv1d(64, 128, kernel_size=5, stride=3, padding=2) #1
        self.bn6 = nn.BatchNorm1d(128)
        '''

        # TODO: more sophisticated network required
        self.final_l = hidden_size_2 * 2#
        self.fc_final = nn.Linear(self.final_l, out_dim)

    def forward(self, x):
        # x (batch, channel, length, type)
        # x = x.type(torch.cuda.FloatTensor)
        # TODO: init of x in data_preprocess
        x = torch.nonzero(x)[:,-1].reshape(x.shape[0:-1])
        x = x.squeeze(dim=1)
        # x = x.reshape(x.shape[0], -1, 3)
        # x = x[:,:,0]*25+x[:,:,-1]*5+x[:,:,2]

        x = self.embedding(x)  # batch * length * hidden=128
        x = x.unsqueeze(dim=1)
        x = self.init_conv(x)
        x = x.squeeze(dim=1)
        x, _ = self.rnn1(x) # b * l * h
        x = x.mean(dim=1) # FIXME: sum(dim=1)
        # x = self.conv1(x.unsqueeze(dim=1)) # b * c=4 * h=64
        # x = self.bn1(x) 
        # x = torch.relu(x)

        # x = self.fc_final(x.reshape(-1, self.final_l)) # b * out_dim
        x = x.reshape(-1, self.final_l)

        return x
        
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
        self.feature_extractor_x = feature_extractor(length, type, 128)
        self.feature_extractor_y = feature_extractor(length, type, 128)

        self.fc_x = nn.Linear(128, 128)
        self.fc_y = nn.Linear(128, 128)

        self.fc_cls = nn.Linear(128, 1)

    def match(self, x, y):
        # x = self.fc_x(x)
        # y = self.fc_y(y)

        # FIXME: the function f(x, y) can vary   *  max  cat 
        # f = torch.max(x, y)
        # f = torch.cat((x,y), dim=-1)
        f = x * y
        # f = torch.relu(f)
        output = self.fc_cls(f)
        output = torch.sigmoid(output.squeeze(-1))
        return output

    def forward(self, x, y):
        x = self.feature_extractor_x(x)
        y = self.feature_extractor_y(y)

        output = self.match(x, y)

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
    
