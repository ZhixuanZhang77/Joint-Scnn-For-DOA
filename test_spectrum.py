# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:50:00 2018

@author: me
"""
import tensorflow.keras.models
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import scipy.signal
import torch.nn as nn
from mode import Model
from model import *

autoencoder= tensorflow.keras.models.load_model('autoencoder.h5')
model_low_liu1 = Model(model_low_liu, flag='DNN')
model_low_liu1.load_model( 'dnn_doa1.pth')
model_low_liu1.model_.eval()

model_low_liu2 = Model(model_low_liu, flag='DNN')
model_low_liu2.load_model( 'dnn_doa2.pth')
model_low_liu2.model_.eval()

model_low_liu3= Model(model_low_liu, flag='DNN')
model_low_liu3.load_model( 'dnn_doa3.pth')
model_low_liu3.model_.eval()

model_low_liu4= Model(model_low_liu, flag='DNN')
model_low_liu4.load_model( 'dnn_doa4.pth')
model_low_liu4.model_.eval()

model_low_liu5= Model(model_low_liu, flag='DNN')
model_low_liu5.load_model( 'dnn_doa5.pth')
model_low_liu5.model_.eval()

model_low_liu6= Model(model_low_liu, flag='DNN')
model_low_liu6.load_model( 'dnn_doa6.pth')
model_low_liu6.model_.eval()

# model_low_liu1= tensorflow.keras.models.load_model( 'DNN_weight/model_low_liu1.h5')
# model_low_liu2= tensorflow.keras.models.load_model( 'DNN_weight/model_low_liu2.h5')
# model_low_liu3= tensorflow.keras.models.load_model(  'DNN_weight/model_low_liu3.h5')
# model_low_liu4= tensorflow.keras.models.load_model( 'DNN_weight/model_low_liu4.h5')
# model_low_liu5= tensorflow.keras.models.load_model( 'DNN_weight/model_low_liu5.h5')
# model_low_liu6= tensorflow.keras.models.load_model( 'DNN_weight/model_low_liu6.h5')


# dnn2018= Model(dnn2018, flag='DNN2018')
# dnn2018.load_model( 'dnn2018.pth')
# dnn2018.model_.eval()
#
# resnet2020= Model(resnet2020, flag='DNN2020')
# resnet2020.load_model( 'resnet2020.pth')
# resnet2020.model_.eval()

cnn_low= Model(cnn, flag='DCN')
cnn_low.load_model( 'dcn_doa.pth')
cnn_low.model_.eval()
#
sresnet7 = Model(sresnet7, flag='DCN')
sresnet7.load_model('sresnet7.pth')
sresnet7.model_.eval()
sresnet7.model_ = sresnet7.model_.to(torch.device('cpu'))


sresnet7_v2 = Model(sresnet7_v2, flag='DCN_v2')
sresnet7_v2.load_model('sresnet7_v2.pth')
sresnet7_v2.model_.eval()
sresnet7_v2.model_ = sresnet7_v2.model_.to(torch.device('cpu'))

K=4#4#2#3#2
k= 2#2#3#2#2

read_temp=scipy.io.loadmat('data/data2_test_s{}.mat'.format(K))



S_est=read_temp['S_est']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
DOA_train=read_temp['DOA_train']
theta=read_temp['theta']
gamma=read_temp['gamma']
gamma_R=read_temp['gamma_R']
S_label1 = np.expand_dims(S_label, 2)

from sklearn import preprocessing 
normalizer = preprocessing.Normalizer().fit(R_est)
[r2,c]=np.shape(R_est) # 5, 120
[r2,I]=np.shape(S_label) # 5, 56
DOA=np.arange(I)-60
print(DOA_train)
# exit()
print('S_est shape: ',S_est.shape)
# DNN2018=np.zeros((I,r2))
# ResNet2020=np.zeros((I,r2))
Lu=np.zeros((I,r2))
DCN=np.zeros((I,r2))
Sresnet7=np.zeros((I,r2))
Sresnet7_v2=np.zeros((I,r2))
for i in range(r2):
    T=R_est[i,:].reshape(1,c)
    Y_autocode_T = autoencoder.predict(T)
    # Y_autocode_T[:,0*c:1*c] = normalizer.transform(Y_autocode_T[:,0*c:1*c])
    # Y_autocode_T[:,1*c:2*c] = normalizer.transform(Y_autocode_T[:,1*c:2*c])
    # Y_autocode_T[:,2*c:3*c] = normalizer.transform(Y_autocode_T[:,2*c:3*c])
    # Y_autocode_T[:,3*c:4*c] = normalizer.transform(Y_autocode_T[:,3*c:4*c])
    # Y_autocode_T[:,4*c:5*c] = normalizer.transform(Y_autocode_T[:,4*c:5*c])
    # Y_autocode_T[:,5*c:6*c] = normalizer.transform(Y_autocode_T[:,5*c:6*c])

    DF_T_low_liu1 = model_low_liu1.model_(
        torch.from_numpy(Y_autocode_T[:, 0 * c:1 * c])).detach().cpu().numpy().tolist()
    DF_T_low_liu2 = model_low_liu2.model_(
        torch.from_numpy(Y_autocode_T[:, 1 * c:2 * c])).detach().cpu().numpy().tolist()
    DF_T_low_liu3 = model_low_liu3.model_(
        torch.from_numpy(Y_autocode_T[:, 2 * c:3 * c])).detach().cpu().numpy().tolist()
    DF_T_low_liu4 = model_low_liu4.model_(
        torch.from_numpy(Y_autocode_T[:, 3 * c:4 * c])).detach().cpu().numpy().tolist()
    DF_T_low_liu5 = model_low_liu5.model_(
        torch.from_numpy(Y_autocode_T[:, 4 * c:5 * c])).detach().cpu().numpy().tolist()
    DF_T_low_liu6 = model_low_liu6.model_(
        torch.from_numpy(Y_autocode_T[:, 5 * c:6 * c])).detach().cpu().numpy().tolist()
    DF_T_liu = [DF_T_low_liu1, DF_T_low_liu2, DF_T_low_liu3
        , DF_T_low_liu4, DF_T_low_liu5, DF_T_low_liu6]
    DF_T_liu=np.array(DF_T_liu)
    DF_T_liu=np.reshape(DF_T_liu,I)
    Lu[:,i]= DF_T_liu

    # DF_T_liu1 = model_low_liu1.predict(Y_autocode_T[:, 0 * c:1 * c])
    # DF_T_liu2 = model_low_liu2.predict(Y_autocode_T[:, 1 * c:2 * c])
    # DF_T_liu3 = model_low_liu3.predict(Y_autocode_T[:, 2 * c:3 * c])
    # DF_T_liu4 = model_low_liu4.predict(Y_autocode_T[:, 3 * c:4 * c])
    # DF_T_liu5 = model_low_liu5.predict(Y_autocode_T[:, 4 * c:5 * c])
    # DF_T_liu6 = model_low_liu6.predict(Y_autocode_T[:, 5 * c:6 * c])
    # DF_T_liu = [DF_T_liu1, DF_T_liu2, DF_T_liu3
    #     , DF_T_liu4, DF_T_liu5, DF_T_liu6]
    # DF_T_liu = np.array(DF_T_liu)
    # DF_T_liu = np.reshape(DF_T_liu, I)
    # Lu[:, i] = DF_T_liu

    # DF_T_dnn2018 = dnn2018.model_(
    #     torch.tensor(S_est[i, :, :].reshape(1, -1), dtype=torch.float)) \
    #     .detach().numpy().tolist()
    # DF_T_dnn2018 = np.array(DF_T_dnn2018)
    # DF_T_dnn2018 = np.reshape(DF_T_dnn2018, I)
    # DNN2018[:, i] = DF_T_dnn2018

    # DF_T_resnet2020 = resnet2020.model_(
    #     torch.tensor(S_est[i, :, :].reshape(1, -1), dtype=torch.float)) \
    #     .detach().numpy().tolist()
    # DF_T_resnet2020 = np.array(DF_T_resnet2020)
    # DF_T_resnet2020 = np.reshape(DF_T_resnet2020, I)
    # ResNet2020[:, i] = DF_T_resnet2020

    DF_T_cnn = cnn_low.model_(
        torch.tensor(S_est[i, :, :].reshape(1, I, 2), dtype=torch.float).permute(0, 2, 1)) \
        .detach().numpy().tolist()
    DF_T_cnn=np.array(DF_T_cnn)    
    DF_T_cnn=np.reshape(DF_T_cnn,I)
    DCN[:,i]=DF_T_cnn

    DF_T_sresnet7 = sresnet7.model_(
        torch.tensor(S_est[i, :, :].reshape(1, I, 2), dtype=torch.float, device=torch.device('cpu')).permute(0, 2, 1)) \
        .detach().numpy().tolist()
    DF_T_sresnet7 = np.array(DF_T_sresnet7)
    DF_T_sresnet7 = np.reshape(DF_T_sresnet7, I)
    Sresnet7[:, i] = DF_T_sresnet7
 
    DF_T_sresnet7_v2 = sresnet7_v2.model_(
        torch.tensor(S_est[i, :, :].reshape(1, I, 2), dtype=torch.float, device=torch.device('cpu')).permute(0, 2, 1))[1] \
        .detach().numpy().tolist()
    DF_T_sresnet7_v2 = np.array(DF_T_sresnet7_v2)
    DF_T_sresnet7_v2 = np.reshape(DF_T_sresnet7_v2, I)
    Sresnet7_v2[:, i] = DF_T_sresnet7_v2

# figsize = 5,4
# figure, ax = plt.subplots(figsize=figsize)
# plt.plot(theta.T,(DNN2018[:,k]),linewidth=2.0)
# plt.ylim([-0.1,1.1])
# plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
# plt.tick_params(labelsize=13)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
# plt.xlabel('DOA($^o$)',font2)
# plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
# plt.title('DNN2018')
# plt.show()

# figsize = 5,4
# figure, ax = plt.subplots(figsize=figsize)
# plt.plot(theta.T,(ResNet2020[:,k]),linewidth=2.0)
# plt.ylim([-0.1,1.1])
# plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
# plt.tick_params(labelsize=13)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 12,}
# plt.xlabel('DOA($^o$)',font2)
# plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
# plt.title('ResNet2020')
# plt.show()

name='DCN-DOA'
figsize = 8,8
figure, ax = plt.subplots(figsize=figsize)
plt.plot(theta.T,(DCN[:,k]),linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
plt.tick_params(labelsize=20)#13
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 30,}#12
plt.xlabel('DOA($^o$)',font2)
plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 40,}#12
plt.title(name,font1)
if K!=2:
    plt.savefig('./fig/k{}/{}_spectrum_k{}.pdf'.format(K,name,K), dpi=200)
else:
    plt.savefig('./fig/k{}/{}_spectrum_k{}-i{}.pdf'.format(K,name,K,k), dpi=200)
plt.show()

name='DNN-DOA'
figsize = 8,8
figure, ax = plt.subplots(figsize=figsize)
plt.plot(theta.T,np.mean(Lu,1),linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
plt.tick_params(labelsize=20)#13
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 30,}#12
plt.xlabel('DOA($^o$)',font2)
plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 40,}#12
plt.title(name,font1)
if K!=2:
    plt.savefig('./fig/k{}/{}_spectrum_k{}.pdf'.format(K,name,K), dpi=200)
else:
    plt.savefig('./fig/k{}/{}_spectrum_k{}-i{}.pdf'.format(K,name,K,k), dpi=200)
plt.show()
# exit()
name='Scnn'
figsize = 8,8
figure, ax = plt.subplots(figsize=figsize)
plt.plot(theta.T,(Sresnet7[:,k]),linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
plt.tick_params(labelsize=20)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 30,}#12
plt.xlabel('DOA($^o$)',font2)
plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 40,}#12
plt.title(name,font1)
if K!=2:
    plt.savefig('./fig/k{}/{}_spectrum_k{}.pdf'.format(K,name,K), dpi=200)
else:
    plt.savefig('./fig/k{}/{}_spectrum_k{}-i{}.pdf'.format(K,name,K,k), dpi=200)
plt.show()

name='Joint-Scnn'
figsize = 8,8
figure, ax = plt.subplots(figsize=figsize)
plt.plot(theta.T,(Sresnet7_v2[:,k]),linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.plot(DOA_train[:,k],np.ones((K,)),'rD')
plt.tick_params(labelsize=20)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 30,}#12
plt.xlabel('DOA($^o$)',font2)
plt.ylabel('Spectrum',font2) #将文件保存至文件中并且画出图
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 40,}#12
plt.title(name,font1)
if K!=2:
    plt.savefig('./fig/k{}/{}_spectrum_k{}.pdf'.format(K,name,K), dpi=200)
else:
    plt.savefig('./fig/k{}/{}_spectrum_k{}-i{}.pdf'.format(K,name,K,k), dpi=200)
plt.show()