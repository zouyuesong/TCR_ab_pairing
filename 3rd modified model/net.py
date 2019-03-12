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

        self.conv1 = nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=5, stride=3, padding=2)
        self.bn3 = nn.BatchNorm1d(16)
        self.conv4 = nn.Conv1d(16,32, kernel_size=5, stride=3, padding=2)
        self.bn4 = nn.BatchNorm1d(32)
        self.conv5 = nn.Conv1d(32,60, kernel_size=5, stride=3, padding=2)
        self.bn5 = nn.BatchNorm1d(60)

        # TODO: more sophisticated network required
        self.final_l = 180 #16*int(length/8)
        self.fc_final = nn.Linear(self.final_l, out_dim)

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
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        
        x = nn.AvgPool1d(2)(x)

        x = self.fc_final(x.reshape(-1, self.final_l))

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
        self.feature_extractor_x = feature_extractor(length, type, 512)
        self.feature_extractor_y = feature_extractor(length, type, 512)

        self.fc_x = nn.Linear(128, 128)
        self.fc_y = nn.Linear(128, 128)

        self.fc_cls = nn.Linear(512*2, 1)


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
        output = torch.sigmoid(output)
        output = torch.cat((output, 1-output), dim=-1)

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
    
