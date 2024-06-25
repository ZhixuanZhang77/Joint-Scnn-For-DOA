import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
from SNN_tools.snn import *
from SNN_tools.functional import ZeroExpandInput_CNN,ZeroExpandInput_CNN_v2

cnn = nn.Sequential(
            nn.Conv1d(2, 12, 25, padding=12),
            nn.ReLU(),
            nn.Conv1d(12, 6, 15, padding=7),
            nn.ReLU(),
            nn.Conv1d(6, 3, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(3, 1, 3, padding=1),
            nn.ReLU(),

)

cnn_tanh = nn.Sequential(
            nn.Conv1d(2, 12, 25, padding=12),
            nn.Tanh(),
            nn.Conv1d(12, 6, 15, padding=7),
            nn.Tanh(),
            nn.Conv1d(6, 3, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(3, 1, 3, padding=1),
            nn.Tanh()
        )

L=120
dnn = nn.Sequential(
    nn.Linear(L*2, int(2*L/3)),
    nn.ReLU(),
    nn.Linear(int(2*L/3), int(4*L/9)),
    nn.ReLU(),
    nn.Linear(int(4*L/9), int(2*L/3)),
    nn.ReLU(),
    nn.Linear(int(2*L/3), L),
    nn.ReLU(),
)

P=6
I=120
t=int(np.floor(I/P))
c=56
model_low_liu = nn.Sequential(
        nn.Linear(c, int(2 * c / 3)),
        nn.Tanh(),
        nn.Linear(int(2 * c / 3), int(4 * c / 9)),
        nn.Tanh(),
        nn.Linear(int(4 * c / 9), t),
        nn.Tanh(),
    )



class ConvBlock(nn.Module):
    def __init__(self, Cin, Cout, kernel, padding):
        super(ConvBlock, self).__init__()
        self.Conv1 = nn.Conv1d(Cin, Cout//2,kernel,padding=padding)
        self.bn1 = nn.BatchNorm1d(Cout//2,eps=1e-25,momentum=0.5)
        self.Conv2 = nn.Conv1d(Cout//2, Cout, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(Cout , eps=1e-25,momentum=0.5)
        self.Conv3 = nn.Conv1d(Cin, Cout, kernel, padding=padding)
        self.bn3 = nn.BatchNorm1d(Cout, eps=1e-25,momentum=0.5)
        # self.pooling = nn.MaxPool1d(2,2)
    def forward(self, x):
        x1 = F.relu(self.bn1(self.Conv1(x)))
        x = F.relu(self.bn2(self.Conv2(x1)))+F.relu(self.bn3(self.Conv3(x)))
        return x


class ConvBlock_v1(nn.Module):
    def __init__(self, Cin, Cout, kernel, padding):
        super(ConvBlock_v1, self).__init__()
        self.Conv1 = nn.Conv1d(Cin, Cout//2,kernel,padding=padding)
        self.bn1 = nn.BatchNorm1d(Cout//2,eps=1e-26,momentum=0.9)
        self.Conv2 = nn.Conv1d(Cout//2, Cout//2,3,padding=1)
        self.bn2 = nn.BatchNorm1d(Cout//2 , eps=1e-26,momentum=0.9)
        self.Conv3 = nn.Conv1d(Cin, Cout//2,1)
        self.bn3 = nn.BatchNorm1d(Cout//2, eps=1e-26,momentum=0.9)
        # self.pooling = nn.MaxPool1d(2,2)
    def forward(self, x):
        x1 = F.relu(self.bn1(self.Conv1(x)))
        x = torch.cat((F.relu(self.bn2(self.Conv2(x1))), F.relu(self.bn3(self.Conv3(x)))), dim=1)
        # x = F.relu(self.bn2(self.Conv2(x1)))+F.relu(self.bn3(self.Conv3(x)))/2
        return x

class Resnet7(nn.Module):
    def __init__(self):
        super(Resnet7, self).__init__()
        self.Block1 = ConvBlock(2, 8, 31, padding=15) #25 12
        self.Block2 = ConvBlock(8, 16, 25, padding=12) #17  8
        self.Block3 = ConvBlock(16, 8, 15, padding=7)  #9   4
        self.output = nn.Sequential(nn.Conv1d(8,1,1),
                                    nn.ReLU()
                                    )

    def forward(self, x):

        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.output(x)
        return x

resnet7 = Resnet7()

# cnn = nn.Sequential(
#             nn.Conv1d(2, 12, 25, padding=12),
#             nn.ReLU(),
#             nn.Conv1d(12, 6, 15, padding=7),
#             nn.ReLU(),
#             nn.Conv1d(6, 3, 5, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(3, 1, 3, padding=1),
#             nn.ReLU(),
#
# )

class Resnet7_v1(nn.Module):
    def __init__(self):
        super(Resnet7_v1, self).__init__()
        self.Block1 = ConvBlock_v1(2, 16, 11, padding=5) #4 9
        self.Block2 = ConvBlock_v1(16,32, 7, padding=3) #60  5
        self.Block3 = ConvBlock_v1(32, 16, 3, padding=1)  #9   4
        self.output = nn.Sequential(nn.Conv1d(16,1,1),
                                   nn.ReLU()
                                    )

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.output(x)
        return x

resnet7_v1 = Resnet7_v1()




class sResnet7(nn.Module):
    def __init__(self, T=10,isShowheatmap=False):   ######################  SCNN   (using original tandem learning)      #####################################
        super(sResnet7, self).__init__()
        # self.Block1 = ConvBlockBN1d(2, 16, 31, padding=15, eps=1e-25, momentum=0.5)#1e-25
        # self.Block2 = ConvBlockBN1d(16, 64, 25, padding=12, eps=1e-25, momentum=0.5)
        # self.Block3 = ConvBlockBN1d(64, 16, 15, padding=7, eps=1e-25, momentum=0.5)
        # self.output = nn.Conv1d(16, 1, 1)
        self.energy = 0
        self.Block1 = ConvBlockBN1d(2, 32, 31, padding=15, eps=1e-25, momentum=0.9)  # 1e-25  , momentum=0.5
        self.Block2 = ConvBlockBN1d(32, 96, 25, padding=12, eps=1e-25, momentum=0.9)
        self.Block3 = ConvBlockBN1d(96, 32, 15, padding=7, eps=1e-25, momentum=0.9)
        self.output = nn.Conv1d(32, 1, 1)
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        nn.init.uniform_(self.output.bias, -0.1, 0.1)
        self.encoder = ZeroExpandInput_CNN.apply
        self.T = T
        self.isShowheatmap = isShowheatmap
        if isShowheatmap:
            self.layer_spikes =[]
    def forward(self, x):
        energy = 0
        x_st, x = self.encoder(x, self.T)
        x_st1, x1 = self.Block1(x_st, x)
        energy += self.Block1.energy
        x_st2, x2 = self.Block2(x_st1, x1)
        energy += self.Block2.energy
        x_st3, x3 = self.Block3(x_st2, x2)
        energy += self.Block3.energy
        energy *=0.9
        x_out = F.relu(self.output(x3))
        energy += (32*120*4.6*x_out.size(0))
        self.energy = energy
        if self.isShowheatmap:
            self.layer_spikes = [x_st1, x_st2, x_st3]
        return x_out

sresnet7 = sResnet7(T=50)#25

class sResnet7_v2(nn.Module):    ###################### Ours: Joint-SCNN   (using our proposed tandem learning)      #####################################
    def __init__(self, T=10,isShowheatmap=False):
        super(sResnet7_v2, self).__init__()
        # self.Block1 = ConvBlockBN1d(2, 16, 31, padding=15, eps=1e-25, momentum=0.5)#1e-25
        # self.Block2 = ConvBlockBN1d(16, 64, 25, padding=12, eps=1e-25, momentum=0.5)
        # self.Block3 = ConvBlockBN1d(64, 16, 15, padding=7, eps=1e-25, momentum=0.5)
        # self.output = nn.Conv1d(16, 1, 1)
        self.energy = 0
        self.Block1 = ConvBlockBN1d_v2(2, 32, 31, padding=15, eps=1e-25, momentum=0.9)  # 1e-25  , momentum=0.9
        self.Block2 = ConvBlockBN1d_v2(32, 96, 25, padding=12, eps=1e-25, momentum=0.9)
        self.Block3 = ConvBlockBN1d_v2(96, 32, 15, padding=7, eps=1e-25, momentum=0.9)
        self.output = ConvBN1d_v4(32, 1, 1, last_layer=True, eps=1e-25, momentum=0.9)#nn.Conv1d(32, 1, 1)
        nn.init.uniform_(self.output.conv1d.weight, -0.1, 0.1)
        nn.init.uniform_(self.output.conv1d.bias, -0.1, 0.1)
        self.encoder = ZeroExpandInput_CNN_v2.apply
        self.T = T
        self.isShowheatmap = isShowheatmap
        if isShowheatmap:
            self.layer_spikes =[]
    def forward(self, x):
        energy = 0
        x,x_st, x_sc = self.encoder(x, self.T)
        x1,x_st1, x_sc1 = self.Block1(x,x_st, x_sc)
        energy += self.Block1.energy
        x2, x_st2, x_sc2 = self.Block2(x1,x_st1, x_sc1)
        energy += self.Block2.energy
        x3, x_st3, x_sc3 = self.Block3(x2, x_st2, x_sc2)
        energy += self.Block3.energy
        energy *=0.9
        x_out,_,x_sc = self.output(x3, x_st3, x_sc3) #
        energy += (32*120*4.6*x_out.size(0))
        self.energy = energy
        if self.isShowheatmap:
            self.layer_spikes = [x_st1, x_st2, x_st3]
        return F.relu(x_out), F.relu(x_sc)

sresnet7_v2 = sResnet7_v2(T=50)#25