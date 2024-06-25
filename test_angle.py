# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:53:53 2019

@author: me
"""

import keras.models
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import scipy.signal
import heapq
import time
import tensorflow
import torch
import torch.nn as nn
from mode import Model
from model import *
def findpeaks(x, K,DOA):
    x=np.array(x)
    indexes,_= scipy.signal.find_peaks(x)    
    maxi=heapq.nlargest(K, x[indexes])
    ind=np.zeros((K,))
    ind=np.int_(ind)
    p=np.zeros((K,))

    if len(indexes)==0:
        ind[0]=60
        ind[1]=59
        ind=np.int_(ind)
        p=DOA[ind]

    else:   
        if len(indexes)<K:
           ind[0]=indexes
           ind[1]=60
           ind=np.int_(ind)
           p=DOA[ind]
        else:
             for i in range(K):
                ind[i]=np.where(x==maxi[i])[0][0]
                ind[i]=np.int_(ind[i])
                if ind[i]==0:
                   p[i]=DOA[ind[i]]    
                else:
                    l=int(ind[i]-1)
                    
                    r=int(ind[i]+1)
                    ind[i]=np.int_(ind[i])                   
                    if x[l]<x[r]:
                         p[i]=x[r]/(x[r]+x[ind[i]])*DOA[r]+x[ind[i]]/(x[r]+x[ind[i]])*DOA[ind[i]]
                    else:
                         p[i]=x[l]/(x[l]+x[ind[i]])*DOA[l]+x[ind[i]]/(x[l]+x[ind[i]])*DOA[ind[i]]
                         
            
     
    ind=np.int_(ind)                          
  
#    return DOA[ind],p
    return p,DOA[ind]


autoencoder= tensorflow.keras.models.load_model('autoencoder.h5')
#
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


cnn_low= Model(cnn, flag='DCN')
cnn_low.load_model( 'dcn_doa.pth')
cnn_low.model_.eval()
#
sresnet7 = Model(sresnet7, flag='DCN')
sresnet7.load_model('sresnet7.pth')
sresnet7.model_.eval()
sresnet7.model_ = sresnet7.model_.to(torch.device('cpu'))
#
sresnet7_v2 = Model(sresnet7_v2, flag='DCN')
sresnet7_v2.load_model('sresnet7_v2.pth')
sresnet7_v2.model_.eval()
sresnet7_v2.model_ = sresnet7_v2.model_.to(torch.device('cpu'))

read_temp=scipy.io.loadmat('data/data2_angle.mat')
S_est=read_temp['S_est']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
DOA_train=read_temp['DOA_train']
theta=read_temp['theta']
gamma=read_temp['gamma']
gamma_R=read_temp['gamma_R']
angle=read_temp['Angle']
T_SBC_R=read_temp['T_SBC_R']
T_SBC=read_temp['T_SBC']

S_label1 = np.expand_dims(S_label, 2)
from sklearn import preprocessing 
normalizer = preprocessing.Normalizer().fit(R_est[:,:,0])

[r2,c,S]=np.shape(R_est)
[r2,I,S]=np.shape(S_label)
K=2

test_liu_low=np.zeros((r2,1))
test_cnn_low=np.zeros((r2,1))
test_sresnet7 = np.zeros((r2,1))
test_sresnet7_v2 = np.zeros((r2,1))

RMSE_liu_low=np.zeros((S,1))
RMSE_cnn_low=np.zeros((S,1))
RMSE_sresnet7 = np.zeros((S,1))
RMSE_sresnet7_v2 = np.zeros((S,1))


DOA=np.arange(I)-60

for j in range(S):
    for i in range(r2):
        print('Angle: {}/{}, {}/{}'.format(j+1,S,i + 1, r2))
        T=R_est[i,:,j].reshape(1,c)
        Y_autocode_T =autoencoder.predict(T,verbose = 0)
        # Y_autocode_T[:,0*c:1*c]=normalizer.transform(Y_autocode_T[:,0*c:1*c])
        # Y_autocode_T[:,1*c:2*c]=normalizer.transform(Y_autocode_T[:,1*c:2*c])
        # Y_autocode_T[:,2*c:3*c]=normalizer.transform(Y_autocode_T[:,2*c:3*c])
        # Y_autocode_T[:,3*c:4*c]=normalizer.transform(Y_autocode_T[:,3*c:4*c])
        # Y_autocode_T[:,4*c:5*c]=normalizer.transform(Y_autocode_T[:,4*c:5*c])
        # Y_autocode_T[:,5*c:6*c]=normalizer.transform(Y_autocode_T[:,5*c:6*c])
        
        DF_T_low_liu1 = model_low_liu1.model_(torch.from_numpy(Y_autocode_T[:,0*c:1*c])).detach().cpu().numpy().tolist()
        DF_T_low_liu2 = model_low_liu2.model_(torch.from_numpy(Y_autocode_T[:,1*c:2*c])).detach().cpu().numpy().tolist()
        DF_T_low_liu3 = model_low_liu3.model_(torch.from_numpy(Y_autocode_T[:,2*c:3*c])).detach().cpu().numpy().tolist()
        DF_T_low_liu4 = model_low_liu4.model_(torch.from_numpy(Y_autocode_T[:,3*c:4*c])).detach().cpu().numpy().tolist()
        DF_T_low_liu5 = model_low_liu5.model_(torch.from_numpy(Y_autocode_T[:,4*c:5*c])).detach().cpu().numpy().tolist()
        DF_T_low_liu6 = model_low_liu6.model_(torch.from_numpy(Y_autocode_T[:,5*c:6*c])).detach().cpu().numpy().tolist()
        DF_T_low_liu = [DF_T_low_liu1,DF_T_low_liu2,DF_T_low_liu3
                          ,DF_T_low_liu4,DF_T_low_liu5,DF_T_low_liu6]
        DF_T_low_liu=np.array(DF_T_low_liu)
        DF_T_low_liu=np.reshape(DF_T_low_liu,I)

        # DF_T_liu1 = model_low_liu1.predict(Y_autocode_T[:, 0 * c:1 * c])
        # DF_T_liu2 = model_low_liu2.predict(Y_autocode_T[:, 1 * c:2 * c])
        # DF_T_liu3 = model_low_liu3.predict(Y_autocode_T[:, 2 * c:3 * c])
        # DF_T_liu4 = model_low_liu4.predict(Y_autocode_T[:, 3 * c:4 * c])
        # DF_T_liu5 = model_low_liu5.predict(Y_autocode_T[:, 4 * c:5 * c])
        # DF_T_liu6 = model_low_liu6.predict(Y_autocode_T[:, 5 * c:6 * c])
        # DF_T_low_liu = [DF_T_liu1, DF_T_liu2, DF_T_liu3
        #     , DF_T_liu4, DF_T_liu5, DF_T_liu6]
        # DF_T_low_liu = np.array(DF_T_low_liu)
        # DF_T_low_liu = np.reshape(DF_T_low_liu, I)


        DF_T_cnn_low=cnn_low.model_(torch.tensor(S_est[i,:,:,j].reshape(1,I,2),dtype=torch.float).permute(0,2,1))\
            .detach().cpu().numpy().tolist()
        DF_T_cnn_low=np.array(DF_T_cnn_low)    
        DF_T_cnn_low=np.reshape(DF_T_cnn_low,I)

        DF_T_sresnet7 = sresnet7.model_(
            torch.tensor(S_est[i, :, :, j].reshape(1, I, 2), dtype=torch.float,device=torch.device('cpu')).permute(0, 2, 1)) \
            .detach().cpu().numpy().tolist()
        DF_T_sresnet7 = np.array(DF_T_sresnet7)
        DF_T_sresnet7 = np.reshape(DF_T_sresnet7, I)

        DF_T_sresnet7_v2 = sresnet7_v2.model_(
            torch.tensor(S_est[i, :, :, j].reshape(1, I, 2), dtype=torch.float, device=torch.device('cpu')).permute(0,2,1))[1] \
            .detach().numpy().tolist()
        DF_T_sresnet7_v2 = np.array(DF_T_sresnet7_v2)
        DF_T_sresnet7_v2 = np.reshape(DF_T_sresnet7_v2, I)

        DOA_liu_low,_=findpeaks(DF_T_low_liu,K,DOA)
        DOA_cnn_low,_=findpeaks(DF_T_cnn_low,K,DOA)
        DOA_sresnet7, _ = findpeaks(DF_T_sresnet7, K, DOA)
        DOA_sresnet7_v2, _ = findpeaks(DF_T_sresnet7_v2, K, DOA)
        
        test_liu_low[i]=np.mean(np.square(np.sort(DOA_liu_low)-DOA_train[:,i,j]))
        test_cnn_low[i]=np.mean(np.square(np.sort(DOA_cnn_low)-DOA_train[:,i,j]))
        test_sresnet7[i] = np.mean(np.square(np.sort(DOA_sresnet7) - DOA_train[:, i, j]))
        test_sresnet7_v2[i] = np.mean(np.square(np.sort(DOA_sresnet7_v2) - DOA_train[:, i, j]))

    RMSE_liu_low[j]=np.sqrt(np.mean(test_liu_low))
    RMSE_cnn_low[j]=np.sqrt(np.mean(test_cnn_low))
    RMSE_sresnet7[j] = np.sqrt(np.mean(test_sresnet7))
    RMSE_sresnet7_v2[j] = np.sqrt(np.mean(test_sresnet7_v2))
    print(j)    




figsize = 10,10
figure, ax = plt.subplots(figsize=figsize)

plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.semilogy(DOA_train[1,1,:]-DOA_train[0,1,:],RMSE_liu_low+0.0*np.ones((S,1)))
plt.semilogy(DOA_train[1,1,:]-DOA_train[0,1,:],RMSE_cnn_low)
plt.semilogy(DOA_train[1,1,:]-DOA_train[0,1,:],RMSE_sresnet7)
plt.semilogy(DOA_train[1,1,:]-DOA_train[0,1,:],RMSE_sresnet7_v2)
plt.legend(['DNN-DOA','DCN-DOA','Scnn','Joint-Scnn'],loc='upper right',fontsize=15)
plt.xlabel('Angle Seperation($^o$)',font2)
plt.ylabel('RMSE($^o$)',font2) #将文件保存至文件中并且画出图
plt.ylim([0.1,100])
plt.savefig('./fig/Angle/Angle.pdf', dpi=200)
plt.show()

