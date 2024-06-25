import torch

from torch.utils.data import TensorDataset,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import scipy.io as io
import numpy as np
import os
import copy

def read_Data(path):
    read_temp = io.loadmat(path)
    S_est = read_temp['S_est']
    S_abs = read_temp['S_abs']
    S_label = read_temp['S_label']
    R_est = read_temp['R_est']
    S_label1 = np.expand_dims(S_label, 2)
    return S_est.tolist(), S_label1.tolist(), S_abs.tolist(), S_label.tolist()

def tolist(a):
    return [e.tolist() for e in a]

class Model(object):
    def __init__(self, model_, flag):
        super(Model, self).__init__()
        self.critention = None
        self.optimizer = None

        self.model_ = model_
        # self.init_layer()
        self.device = None
        self.trainDS_path = './data/data2_trainlow.mat'
        self.epochs = 0
        self.model_type =flag
        if 'DNN' in flag:
            self.flag = True
        else:
            self.flag = False


    def summary(self):
        self.history = {'loss': [], 'val_loss': []}

    def compile(self, loss=None, optimizer_detail=None, device='cpu',lr_scheduler=None):
        self.critention = loss

        optimizer, lr,optim_flag = optimizer_detail
        if optim_flag == 'Adam':
            if self.flag == 'DNN2020':
                self.optimizer = optimizer(self.model_.parameters(), lr=lr,
                                           amsgrad=True,weight_decay=5e-5)  # amsgrad=True, 1e-25
            else:
                self.optimizer = optimizer(self.model_.parameters(), lr=lr,  eps=1e-25,amsgrad=True) #amsgrad=True, 1e-25
        elif optim_flag == 'RMSprop':
            self.optimizer = optimizer(self.model_.parameters(), lr=lr, eps=1e-25, momentum=0.9)# , momentum=0.9, centered=True
        elif optim_flag == 'SGD':
            self.optimizer = optimizer(self.model_.parameters(), lr=lr,  momentum=0.9,nesterov=True,weight_decay=1e-4)
        self.device = torch.device(device)
        self.best_weight = None
        self.lr_scheduler =lr_scheduler
        self.best_optimizer = None

    def fit(self, epochs=10000, batch_size=128, shuffle=True, validation_split=0.2,x=0,y=0):
        train_Data = read_Data(self.trainDS_path)
        self.epochs = epochs
        if (not isinstance(x, np.ndarray)) and (not isinstance(y, np.ndarray)):
            if self.flag:
                Data = train_Data[2:]

                train_ds = TensorDataset(
                    torch.FloatTensor(Data[0][:int((1 - validation_split) * len(Data[0]))]),
                    torch.FloatTensor(Data[1][:int((1 - validation_split) * len(Data[0]))]))
                test_ds = TensorDataset(
                    torch.FloatTensor(Data[0][int((1 - validation_split) * len(Data[0])):]),
                    torch.FloatTensor(Data[1][int((1 - validation_split) * len(Data[0])):]))


            else:
                Data = train_Data[:2]

                train_ds = TensorDataset(
                    torch.FloatTensor(Data[0][:int((1 - validation_split) * len(Data[0]))]).permute(0, 2, 1),
                    torch.FloatTensor(Data[1][:int((1 - validation_split) * len(Data[0]))]).permute(0, 2, 1))
                test_ds = TensorDataset(
                    torch.FloatTensor(Data[0][int((1 - validation_split) * len(Data[0])):]).permute(0, 2, 1),
                    torch.FloatTensor(Data[1][int((1 - validation_split) * len(Data[0])):]).permute(0, 2, 1))
        else:
            x = x.tolist()
            y = y.tolist()
            train_ds = TensorDataset(
                torch.FloatTensor(x[:int((1 - validation_split) * len(x))]),
                torch.FloatTensor(y[:int((1 - validation_split) * len(x))]))
            test_ds = TensorDataset(
                torch.FloatTensor(x[int((1 - validation_split) * len(x)):]),
                torch.FloatTensor(y[int((1 - validation_split) * len(x)):]))

        # scheduler = ReduceLROnPlateau(self.optimizer, factor=0.9, patience=3,verbose=True) # 10 0.5
        # scheduler = StepLR(self.optimizer,step_size=20,gamma=0.7)
        if self.lr_scheduler:
            if self.model_type == 'DCN_v2':
                # scheduler = StepLR(self.optimizer, step_size=40, gamma=0.5,verbose=True)
                # scheduler = ReduceLROnPlateau(self.optimizer,patience=12,eps=1e-25,factor=0.1,verbose=True)#0.1
                scheduler = MultiStepLR(self.optimizer, milestones=[60, 120,280], gamma=0.1)  # 0.4 [20,45,90,200] 0.5
            elif self.model_type == 'DNN_relu':
                scheduler = StepLR(self.optimizer, step_size=30, gamma=0.8,verbose=True)
            else:
                scheduler = MultiStepLR(self.optimizer, milestones=[20, 40, 240], gamma=0.55)  # 0.4 [20,45,90,200] 0.5
            # scheduler = MultiStepLR(self.optimizer,milestones=[45,90,180],gamma=0.5) # 0.4 [20,45,90,200] 0.5
            # scheduler = StepLR(self.optimizer,step_size=10,gamma=0.9) # 0.4 [20,45,90,200] 0.5

        train_batchs = DataLoader(train_ds, batch_size=batch_size, num_workers=0,shuffle=shuffle)
        test_batchs = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
        best_val_loss = 10000000
        self.model_.to(self.device)
        alpha=0.9
        for epoch in range(1, epochs+1):
            loss_sum = 0
            self.model_.train()
            for x, y in train_batchs:
                # print("x: ",x)
                # print("y: ",y.sum(dim=2))
                # print("x shape: ",x.shape)
                # print("y shape: ",y.sum(dim=2).shape)
                # exit()
                self.model_.zero_grad()
                self.optimizer.zero_grad()

                if self.model_type == "DCN_v2":
                    pred_y, pred_y_sc = self.model_(x.to(self.device))

                    loss =(1 - alpha) * self.critention(pred_y, y.to(self.device)) + alpha * self.critention(pred_y_sc, y.to(self.device))

                    train_loss = self.critention(pred_y_sc, y.to(self.device))

                else:
                    pred_y = self.model_(x.to(self.device))
                    loss = self.critention(pred_y, y.to(self.device))
                    train_loss = self.critention(pred_y, y.to(self.device))


                loss.backward()
                self.optimizer.step()
                loss_sum += train_loss.detach().cpu().item()
            loss_ = loss_sum/len(train_batchs)
            self.history['loss'] += [loss_]

            val_loss_sum = 0
            with torch.no_grad():
                self.model_.eval()
                for idx, (x, y) in enumerate(test_batchs):
                    self.model_.zero_grad()
                    self.optimizer.zero_grad()
                    if self.model_type == 'DCN_v2':
                        pred_y,pred_y_sc = self.model_(x.to(self.device))
                        val_loss = self.critention(pred_y_sc, y.to(self.device))
                    else:
                        pred_y = self.model_(x.to(self.device))
                        val_loss = self.critention(pred_y, y.to(self.device))
                    val_loss_sum += val_loss.detach().cpu().item()
                val_loss_ = val_loss_sum / len(test_batchs)
                self.history['val_loss'] += [val_loss_]
                if self.lr_scheduler:
                    if self.model_type=='DCN_v2' or self.model_type=='DNN_relu':
                        # scheduler.step(val_loss_)
                        scheduler.step()

                    else:
                        scheduler.step()
                print('Epoch {}, train loss: {:.4f}, val loss: {:.4f}\n'.format(epoch, loss_, val_loss_))
            weight = self.model_.state_dict()
            optimizer = self.optimizer.state_dict()
            if val_loss_sum < best_val_loss:
                best_val_loss = val_loss_sum
                self.best_weight = copy.deepcopy(weight)
                self.best_optimizer = copy.deepcopy(optimizer)


    def save(self, filename):
        checkpoint = {'model':self.best_weight,
                      'optimizer':self.best_optimizer,
                      'epochs':self.epochs,
                      'history': self.history
                      }
        torch.save(checkpoint, os.path.join('./weight',filename))

    def load_model(self,load_path):
        self.model_.load_state_dict(torch.load(os.path.join('./weight', load_path))['model'])







