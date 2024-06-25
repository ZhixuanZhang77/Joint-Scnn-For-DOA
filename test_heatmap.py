from snn_heatmap import snn_heatmap
from mode import Model
from model import *
import torch
from mode import read_Data
from torch.utils.data import TensorDataset,DataLoader
device = torch.device('cuda:0')
validation_split=0.2
batch_size = 1
train_Data = read_Data('./data/data2_trainlow.mat')
Data = train_Data[:2]
test_ds = TensorDataset(
                    torch.FloatTensor(Data[0][int((1 - validation_split) * len(Data[0])):]).permute(0, 2, 1),
                    torch.FloatTensor(Data[1][int((1 - validation_split) * len(Data[0])):]).permute(0, 2, 1))
plot_model ='joint-scnn'#joint-


if plot_model == 'scnn':
    sresnet7 = sResnet7(T=50,isShowheatmap=True)
    sresnet7 = Model(sresnet7, flag='DCN_')
    sresnet7.load_model('sresnet7.pth')
elif plot_model=='joint-scnn':
    sresnet7 = sResnet7_v2(T=50, isShowheatmap=True)
    sresnet7 = Model(sresnet7, flag='DCN_v2')
    sresnet7.load_model('sresnet7_v2.pth')

sresnet7.model_.eval()
sresnet7.model_ = sresnet7.model_.to(device)
test_batchs = DataLoader(test_ds, batch_size=batch_size, num_workers=0)


for idx, (x, y) in enumerate(test_batchs):
    sresnet7.model_.zero_grad()
    print()
    pred_y = sresnet7.model_(x.to(device))
    layer_spikes = sresnet7.model_.layer_spikes
    # print(layer_spikes)
    for l, l_spike in enumerate(layer_spikes):
        print(l)
        # print(l_spike.shape)
        snn_heatmap(l_spike.squeeze(0),l,plot_model)
    exit()
    # print('-----')