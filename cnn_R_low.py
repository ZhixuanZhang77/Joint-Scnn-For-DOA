

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:18:42 2019

@author: me
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

from mode import Model
import torch.nn as nn
from torch.optim import *
from model import *
nb_epoch=300
batch_size=64




# cnn = Model(cnn, flag='DCN')
# cnn.compile(loss=nn.MSELoss(), optimizer_detail=(Adam,1e-3, 'Adam'), device='cuda:0')
# # 6e-4
# cnn.summary()
# cnn.fit(epochs=nb_epoch, batch_size=batch_size)
# cnn.save('dcn_doa.pth')



# cnn_tanh = Model(cnn_tanh, flag='DCN')
# cnn_tanh.compile(loss=nn.MSELoss(), optimizer_detail=(Adam, 1e-3, 'Adam'), device='cuda:0')
# cnn_tanh.summary()
# cnn_tanh.fit(epochs=nb_epoch, batch_size=batch_size)
# cnn_tanh.save('dcn_tanh.pth')





# #
# #
#
#
#

#
# dnn_relu = Model(dnn, flag='DNN')
# dnn_relu.compile(loss=nn.MSELoss(), optimizer_detail=(Adam, 5e-4, 'Adam'), device='cuda:0')
# dnn_relu.summary()
# dnn_relu.fit(epochs=nb_epoch, batch_size=batch_size)
# dnn_relu.save('dnn_relu.pth')



# from keras.models import Model #泛型模型
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# import numpy as np
# import scipy.io
# import keras.layers
# from keras.layers import Dense

# read_temp=scipy.io.loadmat('./data/data2_trainlow.mat')
# S_est=read_temp['S_est']
# S_abs=read_temp['S_abs']
# S_label=read_temp['S_label']
# R_est=read_temp['R_est']
# S_label1 = np.expand_dims(S_label, 2)
# [Sample,L,dim]=np.shape(S_est)
# [r2,c]=np.shape(R_est)
# P=6
# I=120
# t=int(np.floor(I/P))
# autoencoder= keras.models.load_model( 'autoencoder.h5')
# # print(autoencoder)
# # exit()
# # print(c,t) #56 20
# # exit()
# from sklearn import preprocessing
# print(R_est.shape)
# normalizer = preprocessing.Normalizer().fit(R_est)
# Y_autocode_filter=autoencoder.predict(R_est)
# exit()
# def init_layer(x):
#     nn.init.uniform_(x[0].weight.data,-1,1)
#     nn.init.uniform_(x[0].bias.data, -1, 1)
#     nn.init.uniform_(x[2].weight.data, -1, 1)
#     nn.init.uniform_(x[2].bias.data, -1, 1)
#     nn.init.uniform_(x[4].weight.data, -0.1, 0.1)
#     nn.init.uniform_(x[4].bias.data, -0.1, 0.1)
#

# for idx in range(6):#
#     print('training {} classification'.format(idx))
#     # Y_autocode_filter[:, idx * c:(idx+1) * c] = normalizer.transform(Y_autocode_filter[:, idx * c:(idx+1) * c])
#
#     model_low_liu = nn.Sequential(
#
#         nn.Linear(c, int(2 * c / 3)),  #56 37
#         nn.Tanh(),
#         nn.Linear(int(2 * c / 3), int(4 * c / 9)), # 37  24
#         nn.Tanh(),
#         nn.Linear(int(4 * c / 9), t),  #  24    20
#         nn.Tanh(),
#     )
#     init_layer(model_low_liu)
#     model_low_liu = Model(model_low_liu, flag='DNN')
#     model_low_liu.compile(loss=nn.MSELoss(), optimizer_detail=(RMSprop, 1e-4, 'RMSprop'))# 1e-3,'RMSprop'
#     model_low_liu.summary()
#     model_low_liu.fit(x=Y_autocode_filter[:,idx * c:(idx+1) * c], y=S_label[:,idx * t:(idx+1) * t],epochs=nb_epoch
#                                 , batch_size=batch_size,shuffle=True,validation_split=0.2)
#
#     model_low_liu.save('dnn_doa{}.pth'.format(idx+1))


# resnet7 = Model(resnet7, flag='DCN')
# resnet7.compile(loss=nn.MSELoss(), optimizer_detail=(Adam, 5e-4, 'Adam'), device='cuda:0')
# resnet7.summary()
# resnet7.fit(epochs=nb_epoch, batch_size=batch_size)
# resnet7.save('resnet7.pth')

# resnet7 = Model(resnet7_v1, flag='DCN')
# resnet7.compile(loss=nn.MSELoss(), optimizer_detail=(Adam, 1e-3, 'Adam'), device='cuda:0')
# resnet7.summary()
# resnet7.fit(epochs=nb_epoch, batch_size=batch_size)
# resnet7.save('resnet7_v1.pth')

# sresnet7 = Model(sresnet7, flag='DCN')
# sresnet7.compile(loss=nn.MSELoss(), optimizer_detail=(Adam, 1e-3, 'Adam'), device='cuda:0',lr_scheduler=True)#5e-4
# sresnet7.summary()
# sresnet7.fit(epochs=nb_epoch, batch_size=batch_size)
# sresnet7.save('sresnet7.pth')


sresnet7_v2 = Model(sresnet7_v2, flag='DCN_v2')
sresnet7_v2.compile(loss=nn.MSELoss(), optimizer_detail=(Adam, 1e-3, 'Adam'), device='cuda:0',lr_scheduler=True)#5e-4
sresnet7_v2.summary()
sresnet7_v2.fit(epochs=nb_epoch, batch_size=batch_size)
sresnet7_v2.save('sresnet7_v2.pth')


