from snn_heatmap import snn_heatmap
from mode import Model
from model import *
import torch
from mode import read_Data
from torch.utils.data import TensorDataset,DataLoader
device = torch.device('cuda:0')

model_nm = 'sresnet7_v2'
validation_split=0.2
batch_size = 1
train_Data = read_Data('./data/data2_trainlow.mat')
Data = train_Data[:2]
test_ds = TensorDataset(
                    torch.FloatTensor(Data[0][int((1 - validation_split) * len(Data[0])):]).permute(0, 2, 1),
                    torch.FloatTensor(Data[1][int((1 - validation_split) * len(Data[0])):]).permute(0, 2, 1))
if model_nm == 'sresnet7_v2':
    sresnet7_v2 = sResnet7_v2(T=50,isShowheatmap=True)
    sresnet7 = Model(sresnet7_v2, flag='DCN_v2')
    sresnet7.load_model('sresnet7_v2.pth')
elif model_nm == 'sresnet7':
    sresnet7 = sResnet7(T=50, isShowheatmap=True)
    sresnet7 = Model(sresnet7, flag='DCN')
    sresnet7.load_model('sresnet7.pth')

sresnet7.model_.eval()
sresnet7.model_ = sresnet7.model_.to(device)
test_batchs = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

energy = 0
for idx, (x, y) in enumerate(test_batchs):
    sresnet7.model_.zero_grad()
    print(len(test_batchs),idx+1)
    pred_y = sresnet7.model_(x.to(device))

    energy += sresnet7.model_.energy
print(energy/len(test_ds))
print('-----')