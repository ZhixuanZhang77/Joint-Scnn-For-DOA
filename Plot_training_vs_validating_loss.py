

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:18:42 2019

@author: me
"""
import torch

import matplotlib.pyplot as plt 

import numpy as np
import tensorflow as tf
# dnn2018_record = torch.load('./weight/dnn2018.pth')
# resnet2020_record = torch.load('./weight/resnet2020.pth')
dcn_doa_record = torch.load('./weight/dcn_doa.pth')
dcn_tanh_record = torch.load('./weight/dcn_tanh.pth')
# dnn_relu_record = torch.load('./weight/dnn_relu.pth')
srestnet7_record = torch.load('./weight/sresnet7.pth')
srestnet7_v2_record = torch.load('./weight/sresnet7_v2.pth')

# print(min(dcn_doa_record['history']['loss']*1000))
# print(min(dcn_doa_record['history']['val_loss']*1000))
# exit()

loss=[]
val_loss = []
for i in  range(1,7):
     dnn_doa_ = torch.load('./weight/dnn_doa{}.pth'.format(i))
     loss.append(dnn_doa_['history']['loss'])
     val_loss.append(dnn_doa_['history']['val_loss'])
#
loss = np.array(loss).mean(0)
val_loss = np.array(val_loss).mean(0)
dnn_doa_record = {'history':{'loss':loss.tolist(),'val_loss':val_loss.tolist()}}

dnn_relu_record = np.load('./weight/DNN_relu_history.npy',allow_pickle=True,).tolist()

# print(srestnet7_record['history']['loss'][50:70])
# print(np.argwhere(np.array(srestnet7_record['history']['val_loss'])<0.0095))
# exit()

figsize = 8,5
figure, ax = plt.subplots(figsize=figsize)
# plt.plot(np.array(dnn2018_record['history']['val_loss'])*1000)
# plt.plot(np.array(resnet2020_record['history']['val_loss'])*1000)
plt.plot(np.array(dcn_doa_record['history']['val_loss'])*1000)
plt.plot(np.array(dcn_tanh_record['history']['val_loss'])*1000)
plt.plot(np.array(dnn_doa_record['history']['val_loss'])*1000)
plt.plot(np.array(dnn_relu_record['history']['val_loss'])*1000)
plt.plot(np.array(srestnet7_record['history']['val_loss'])*1000)
plt.plot(np.array(srestnet7_v2_record['history']['val_loss'])*1000)

plt.legend(['DCN-DoA','DCN-tanh','DNN-DoA','DNN-relu','Scnn','Joint-Scnn'])
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 13,}
plt.xlabel('Epoch',font2)
plt.ylabel('Test MSE(*1e$^-$$^3$)',font2)
plt.ylim([8,20])
plt.savefig('./fig/training_val_loss/test.pdf', dpi=200)
plt.show()


figsize = 8,5
figure, ax = plt.subplots(figsize=figsize)
# plt.plot(np.array(dnn2018_record['history']['loss'])*1000)
# plt.plot(np.array(resnet2020_record['history']['loss'])*1000)
plt.plot(np.array(dcn_doa_record['history']['loss'])*1000)
plt.plot(np.array(dcn_tanh_record['history']['loss'])*1000)
plt.plot(np.array(dnn_doa_record['history']['loss'])*1000)
plt.plot(np.array(dnn_relu_record['history']['loss'])*1000)
plt.plot(np.array(srestnet7_record['history']['loss'])*1000)
plt.plot(np.array(srestnet7_v2_record['history']['loss'])*1000)

plt.legend(['DCN-DoA','DCN-tanh','DNN-DoA','DNN-relu','Scnn','Joint-Scnn'])#'DNN2018','ResNet2020',
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 13,}
plt.xlabel('Epoch',font2)
plt.ylabel('Train MSE(*1e$^-$$^3$)',font2)
plt.ylim([8,20])
plt.savefig('./fig/training_val_loss/training.pdf', dpi=200)
plt.show()
