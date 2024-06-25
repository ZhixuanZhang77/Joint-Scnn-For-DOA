# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:11:24 2019

@author: me
"""

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
import heapq
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
           ind[1]=indexes+1
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


autoencoder= tensorflow.keras.models.load_model( 'autoencoder.h5')
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

sresnet7_v2 = Model(sresnet7_v2, flag='DCN')
sresnet7_v2.load_model('sresnet7_v2.pth')
sresnet7_v2.model_.eval()
sresnet7_v2.model_ = sresnet7_v2.model_.to(torch.device('cpu'))

read_temp=scipy.io.loadmat('data/data2_testall.mat')
S_est=read_temp['S_est']
S_label=read_temp['S_label']
R_est=read_temp['R_est']
DOA_train=read_temp['DOA_train']
# print(DOA_train.T)
# print(DOA_train.T.shape)

# print(R_est.shape)
# exit()
gamma=read_temp['gamma']
gamma_R=read_temp['gamma_R']
S_label1 = np.expand_dims(S_label, 2)
from sklearn import preprocessing 
normalizer = preprocessing.Normalizer().fit(R_est)
[r2,c]=np.shape(R_est)
[r2,I]=np.shape(S_label)
DOA=np.arange(I)-60

K=2

Lu=np.zeros((I,r2))
DCN=np.zeros((I,r2))
Sresnet7=np.zeros((I,r2))
Sresnet7_v2=np.zeros((I,r2))

test_liu=np.zeros((K,r2))
test_cnn=np.zeros((K,r2))
test_sresnet7=np.zeros((K,r2))
test_sresnet7_v2=np.zeros((K,r2))

for i in range(r2):
    T=R_est[i,:].reshape(1,c)
    Y_autocode_T =autoencoder.predict(T)
    # Y_autocode_T[:,0*c:1*c]=normalizer.transform(Y_autocode_T[:,0*c:1*c])
    # Y_autocode_T[:,1*c:2*c]=normalizer.transform(Y_autocode_T[:,1*c:2*c])
    # Y_autocode_T[:,2*c:3*c]=normalizer.transform(Y_autocode_T[:,2*c:3*c])
    # Y_autocode_T[:,3*c:4*c]=normalizer.transform(Y_autocode_T[:,3*c:4*c])
    # Y_autocode_T[:,4*c:5*c]=normalizer.transform(Y_autocode_T[:,4*c:5*c])
    # Y_autocode_T[:,5*c:6*c]=normalizer.transform(Y_autocode_T[:,5*c:6*c])

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


    DF_T_cnn = cnn_low.model_(
        torch.tensor(S_est[i, :, :].reshape(1, I, 2), dtype=torch.float).permute(0, 2, 1)) \
        .detach().numpy().tolist()
    DF_T_cnn = np.array(DF_T_cnn)
    DF_T_cnn = np.reshape(DF_T_cnn, I)
    DCN[:, i] = DF_T_cnn

    DF_T_sresnet7 = sresnet7.model_(
        torch.tensor(S_est[i, :, :].reshape(1, I, 2), dtype=torch.float, device=torch.device('cpu')).permute(0, 2,
                                                                                                                1)) \
        .detach().numpy().tolist()
    DF_T_sresnet7 = np.array(DF_T_sresnet7)
    DF_T_sresnet7 = np.reshape(DF_T_sresnet7, I)
    Sresnet7[:, i] = DF_T_sresnet7

    DF_T_sresnet7_v2 = sresnet7_v2.model_(
        torch.tensor(S_est[i, :, :].reshape(1, I, 2), dtype=torch.float, device=torch.device('cpu')).permute(0, 2,
                                                                                                             1))[1] \
        .detach().numpy().tolist()
    DF_T_sresnet7_v2 = np.array(DF_T_sresnet7_v2)
    DF_T_sresnet7_v2 = np.reshape(DF_T_sresnet7_v2, I)
    Sresnet7_v2[:, i] = DF_T_sresnet7_v2

    ind_liu,_=findpeaks(DF_T_liu,K,DOA)
    ind_cnn,_=findpeaks(DF_T_cnn,K,DOA)
    ind_sresnet7, _ = findpeaks(DF_T_sresnet7, K, DOA)
    ind_sresnet7_v2, _ = findpeaks(DF_T_sresnet7_v2, K, DOA)

    DOA_cnn=ind_cnn
    DOA_liu=ind_liu
    DOA_sresnet7 = ind_sresnet7
    DOA_sresnet7_v2 = ind_sresnet7_v2

    test_liu[:,i]=np.sort(DOA_liu)
    test_cnn[:,i]=np.sort(DOA_cnn)
    test_sresnet7[:, i] = np.sort(DOA_sresnet7)
    test_sresnet7_v2[:, i] = np.sort(DOA_sresnet7_v2)

name='DNN-DOA'
figsize = 7,7#6,5
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_liu.T[:,0],'.',label='Predicted DOA1')
plt.plot(test_liu.T[:,1],'.',label='Predicted DOA2')
plt.plot(DOA_train.T[:,0],label='DOA1')
plt.plot(DOA_train.T[:,1],label='DOA2')
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.xlabel('Sample index',font2)
plt.ylabel('DOA Estiamte($^o$)',font2) #将文件保存至文件中并且画出图
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.title(name,font1)
plt.legend(loc='upper left',fontsize=15)
plt.savefig('./fig/spectrumall/{}_spectrumall.pdf'.format(name), dpi=200)
plt.show()
# exit()

name='DCN-DOA'
figsize = 7,7
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_cnn.T[:,0],'.',label='Predicted DOA1')
plt.plot(test_cnn.T[:,1],'.',label='Predicted DOA2')
plt.plot(DOA_train.T[:,0],label='DOA1')
plt.plot(DOA_train.T[:,1],label='DOA2')
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.xlabel('Sample index',font2)
plt.ylabel('DOA Estiamte($^o$)',font2)
plt.title(name,font1)
plt.legend(loc='upper left',fontsize=15)
plt.savefig('./fig/spectrumall/{}_spectrumall.pdf'.format(name), dpi=200)
plt.show()

name='Scnn'
figsize = 7,7#6,5
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_sresnet7.T[:,0],'.',label='Predicted DOA1')
plt.plot(test_sresnet7.T[:,1],'.',label='Predicted DOA2')
plt.plot(DOA_train.T[:,0],label='DOA1')
plt.plot(DOA_train.T[:,1],label='DOA2')
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.xlabel('Sample index',font2)
plt.ylabel('DOA Estiamte($^o$)',font2)
plt.title(name,font1)
plt.legend(loc='upper left',fontsize=15)
plt.savefig('./fig/spectrumall/{}_spectrumall.pdf'.format(name), dpi=200)
plt.show()

name='Joint-Scnn'
figsize = 7,7#6,5
figure, ax = plt.subplots(figsize=figsize)
plt.plot(test_sresnet7_v2.T[:,0],'.',label='Predicted DOA1')
plt.plot(test_sresnet7_v2.T[:,1],'.',label='Predicted DOA2')
plt.plot(DOA_train.T[:,0],label='DOA1')
plt.plot(DOA_train.T[:,1],label='DOA2')
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.xlabel('Sample index',font2)
plt.ylabel('DOA Estiamte($^o$)',font2)
plt.title(name,font1)
plt.legend(loc='upper left',fontsize=15)
plt.savefig('./fig/spectrumall/{}_spectrumall.pdf'.format(name), dpi=200)
plt.show()


name='DNN-DOA'
figsize = 7,7#6,5
figure, ax = plt.subplots(figsize=figsize)
err = test_liu.T-DOA_train.T
plt.plot(err[:,0],'.',label='err-DOA1')
plt.plot(err[:,1],'.',label='err-DOA2')
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
plt.ylim([-20,20])
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.xlabel('Sample index',font2)
plt.ylabel('Estiamtion Error($^o$)',font2) #将文件保存至文件中并且画出图
plt.title('err-'+name,font1)
plt.legend(loc='upper right',fontsize=15)
plt.savefig('./fig/err/{}_err.pdf'.format(name), dpi=200)
plt.show()

name='DCN-DOA'
figsize = 7,7#6,5
figure, ax = plt.subplots(figsize=figsize)
err = test_cnn.T-DOA_train.T
plt.plot(err[:,0],'.',label='err-DOA1')
plt.plot(err[:,1],'.',label='err-DOA2')
plt.tick_params(labelsize=15)
plt.ylim([-20,20])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.xlabel('Sample index',font2)
plt.ylabel('Estiamtion Error($^o$)',font2)
plt.title('err-'+name,font1)
plt.legend(loc='upper right',fontsize=15)
plt.savefig('./fig/err/{}_err.pdf'.format(name), dpi=200)
plt.show()

name='Scnn'
figsize = 7,7#6,5
figure, ax = plt.subplots(figsize=figsize)
err=test_sresnet7.T-DOA_train.T
plt.plot(err[:,0],'.',label='err-DOA1')
plt.plot(err[:,1],'.',label='err-DOA2')
plt.tick_params(labelsize=15)
plt.ylim([-20,20])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.xlabel('Sample index',font2)
plt.ylabel('Estiamtion Error($^o$)',font2)
plt.title('err-'+name,font1)
plt.legend(loc='upper right',fontsize=15)
plt.savefig('./fig/err/{}_err.pdf'.format(name), dpi=200)
plt.show()

name='Joint-Scnn'
figsize = 7,7#6,5
figure, ax = plt.subplots(figsize=figsize)
err=test_sresnet7_v2.T-DOA_train.T
plt.plot(err[:,0],'.',label='err-DOA1')
plt.plot(err[:,1],'.',label='err-DOA2')
plt.tick_params(labelsize=15)
plt.ylim([-20,20])
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
font1 = {'family' : 'Times New Roman','weight' : 'normal','size': 20,}
plt.xlabel('Sample index',font2)
plt.ylabel('Estiamtion Error($^o$)',font2)
plt.title('err-'+name,font1)
plt.legend(loc='upper right',fontsize=15)
plt.savefig('./fig/err/{}_err.pdf'.format(name), dpi=200)
plt.show()