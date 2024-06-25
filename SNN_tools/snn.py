import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
from settings import SETTINGS
from SNN_tools.functional import LinearIF, AvgPool2d_IF, Conv2dIF, activate, ConvTranspose2dIF, Conv1dIF,LinearIF_v2, ConvTranspose1dIF


class ConvTransposeBN2d(nn.Module):
    """
	W
	"""

    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0,
                 output_padding=0, bias=True, weight_init=2.0, pooling=1):
        super(ConvTransposeBN2d, self).__init__()
        self.convTranspose2dIF = ConvTranspose2dIF.apply
        self.convTranspose2d = torch.nn.ConvTranspose2d(Cin, Cout, kernel_size, stride, padding, output_padding,
                                                        bias=bias)
        self.bn2d = torch.nn.BatchNorm2d(Cout, eps=1e-4, momentum=0.9)
        # self.bn2d = torch.nn.BatchNorm2d(Cout)
        self.device = device
        self.stride = stride
        self.padding = padding
        self.pooling = pooling
        self.output_padding = output_padding
        nn.init.normal_(self.bn2d.weight, 0, weight_init)

    def forward(self, input_feature_st, input_features_sc):
        # weight update based on the surrogate conv2d layer
        output_bn = F.max_pool2d(self.bn2d(self.convTranspose2d(input_features_sc)), self.pooling)
        output = F.relu(output_bn)
        # output = torch.clamp(output_bn, min=0, max=T)

        # extract the weight and bias from the surrogate conv layer
        convTranspose2d_weight = self.convTranspose2d.weight  # .detach().to(self.device)
        convTranspose2d_bias = self.convTranspose2d.bias  # .detach().to(self.device)
        bnGamma = self.bn2d.weight
        bnBeta = self.bn2d.bias
        bnMean = self.bn2d.running_mean
        bnVar = self.bn2d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights

        ratio = torch.div(bnGamma, torch.sqrt(bnVar))

        weightNorm = torch.mul(convTranspose2d_weight.permute(0, 2, 3, 1), ratio).permute(0, 3, 1, 2)
        biasNorm = torch.mul(convTranspose2d_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the IF layer to get actual output
        # spike train
        output_features_st, output_features_sc = self.convTranspose2dIF(input_feature_st, output, \
                                                                        weightNorm, self.device, biasNorm, \
                                                                        self.stride, self.padding, self.output_padding,
                                                                        self.pooling)

        return output_features_st, output_features_sc


class ConvTransposeBN1d(nn.Module):
    """
	W
	"""

    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0,
                 output_padding=0, bias=True, weight_init=2.0, pooling=1, eps=1e-4, momentum=0.9):
        super(ConvTransposeBN1d, self).__init__()
        self.convTranspose1dIF = ConvTranspose1dIF.apply
        self.convTranspose1d = torch.nn.ConvTranspose1d(Cin, Cout, kernel_size, stride, padding, output_padding,
                                                        bias=bias)
        self.bn1d = torch.nn.BatchNorm1d(Cout, eps=eps, momentum=momentum)
        # self.bn2d = torch.nn.BatchNorm2d(Cout)
        self.device = device
        self.stride = stride
        self.padding = padding
        self.pooling = pooling
        self.output_padding = output_padding
        nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
        nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

    def forward(self, input_feature_st, input_features_sc):
        # weight update based on the surrogate conv2d layer
        output_bn = F.avg_pool1d(self.bn1d(self.convTranspose1d(input_features_sc)), self.pooling)
        output = F.relu(output_bn)
        # output = torch.clamp(output_bn, min=0, max=T)

        # extract the weight and bias from the surrogate conv layer
        convTranspose1d_weight = self.convTranspose1d.weight  # .detach().to(self.device)
        convTranspose1d_bias = self.convTranspose1d.bias  # .detach().to(self.device)
        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights

        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        # in out *kernel
        weightNorm = torch.mul(convTranspose1d_weight.permute(0, 2, 1), ratio).permute(0, 2, 1)
        biasNorm = torch.mul(convTranspose1d_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the IF layer to get actual output
        # spike train
        output_features_st, output_features_sc = self.convTranspose1dIF(input_feature_st, output, \
                                                                        weightNorm, self.device, biasNorm, \
                                                                        self.stride, self.padding, self.output_padding,
                                                                        self.pooling)

        return output_features_st, output_features_sc

class ConvTransposeBN1d_v2(nn.Module):
    """
	W
	"""

    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0,
                 output_padding=0, bias=True, weight_init=2.0, pooling=1, eps=1e-4, momentum=0.9):
        super(ConvTransposeBN1d_v2, self).__init__()
        self.convTranspose1dIF = ConvTranspose1dIF.apply
        self.convTranspose1d = torch.nn.ConvTranspose1d(Cin, Cout, kernel_size, stride, padding, output_padding,
                                                        bias=bias)
        self.bn1d = torch.nn.BatchNorm1d(Cout, eps=eps, momentum=momentum)
        # self.bn2d = torch.nn.BatchNorm2d(Cout)
        self.device = device
        self.stride = stride
        self.padding = padding
        self.pooling = pooling
        self.output_padding = output_padding
        nn.init.uniform_(self.convTranspose1d.weight, -0.5, 0.5)
        nn.init.uniform_(self.convTranspose1d.bias, -0.5, 0.5)

    def forward(self, x, input_feature_st, input_features_sc):
        # weight update based on the surrogate conv2d layer
        output_bn = F.avg_pool1d(self.bn1d(self.convTranspose1d(input_features_sc)), self.pooling)
        output = F.relu(output_bn)
        # output = torch.clamp(output_bn, min=0, max=T)
        x = F.relu(self.bn1d(self.convTranspose1d(x)))
        # extract the weight and bias from the surrogate conv layer
        convTranspose1d_weight = self.convTranspose1d.weight  # .detach().to(self.device)
        convTranspose1d_bias = self.convTranspose1d.bias  # .detach().to(self.device)
        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights

        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        # in out *kernel
        weightNorm = torch.mul(convTranspose1d_weight.permute(0, 2, 1), ratio).permute(0, 2, 1)
        biasNorm = torch.mul(convTranspose1d_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the IF layer to get actual output
        # spike train
        output_features_st, output_features_sc = self.convTranspose1dIF(input_feature_st, output, \
                                                                        weightNorm, self.device, biasNorm, \
                                                                        self.stride, self.padding, self.output_padding,
                                                                        self.pooling)

        return x, output_features_st, output_features_sc



class LinearBN1d(nn.Module):

    def __init__(self, D_in, D_out, device=torch.device(SETTINGS.training.device), bias=True):
        super(LinearBN1d, self).__init__()
        self.linearif = LinearIF.apply
        self.device = device
        self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
        # self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-4, momentum=0.9)
        self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

    # nn.init.normal_(self.bn1d.weight, 0, 2.0)

    def forward(self, input_feature_st, input_features_sc):
        # weight update based on the surrogate linear layer
        T = input_feature_st.shape[1]
        # print(input_features_sc.shape)
        output_bn = self.bn1d(self.linear(input_features_sc))
        output = F.relu(output_bn)

        # extract the weight and bias from the surrogate linear layer
        linearif_weight = self.linear.weight  # .detach().to(self.device)
        linearif_bias = self.linear.bias  # .detach().to(self.device)

        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Linear' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0)
        biasNorm = torch.mul(linearif_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the linearIF layer to get actual output
        # spike train
        output_st, output_sc = self.linearif(input_feature_st, output, weightNorm, \
                                             self.device, biasNorm)

        return output_st, output_sc


class ConvBN2d(nn.Module):
    """
	W
	"""

    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0,
                 bias=True, weight_init=2.0, pooling=1):
        super(ConvBN2d, self).__init__()
        self.conv2dIF = Conv2dIF.apply
        self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
        self.bn2d = torch.nn.BatchNorm2d(Cout, eps=1e-4, momentum=0.9)
        # self.bn2d = torch.nn.BatchNorm2d(Cout)
        self.device = device
        self.stride = stride
        self.padding = padding
        self.pooling = pooling
        nn.init.normal_(self.bn2d.weight, 0, weight_init)

    def forward(self, input_feature_st, input_features_sc):
        T = input_feature_st.shape[1]
        self.conv2d(input_features_sc)
        # weight update based on the surrogate conv2d layer
        output_bn = F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)), self.pooling)
        output = F.relu(output_bn)
        # output = torch.clamp(output_bn, min=0, max=T)

        # extract the weight and bias from the surrogate conv layer
        conv2d_weight = self.conv2d.weight.detach().to(self.device)
        conv2d_bias = self.conv2d.bias.detach().to(self.device)
        bnGamma = self.bn2d.weight
        bnBeta = self.bn2d.bias
        bnMean = self.bn2d.running_mean
        bnVar = self.bn2d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))

        weightNorm = torch.mul(conv2d_weight.permute(1, 2, 3, 0), ratio).permute(3, 0, 1, 2)
        biasNorm = torch.mul(conv2d_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the IF layer to get actual output
        # spike train
        output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output, \
                                                               weightNorm, self.device, biasNorm, \
                                                               self.stride, self.padding, self.pooling)

        return output_features_st, output_features_sc


class sDropout(nn.Module):
    def __init__(self, layerType, pDrop):
        super(sDropout, self).__init__()

        self.pKeep = 1 - pDrop
        self.type = layerType  # 1: Linear 2: Conv

    def forward(self, x_st, x_sc):
        if self.training:
            T = x_st.shape[1]
            mask = torch.bernoulli(x_sc.data.new(x_sc.data.size()).fill_(self.pKeep)) / self.pKeep
            x_sc_out = x_sc * mask
            x_st_out = torch.zeros_like(x_st)

            for t in range(T):
                # Linear Layer
                if self.type == 1:
                    x_st_out[:, t, :] = x_st[:, t, :] * mask
                # Conv1D Layer
                elif self.type == 2:
                    x_st_out[:, t, :, :] = x_st[:, t, :, :] * mask
                # Conv2D Layer
                elif self.type == 3:
                    x_st_out[:, t, :, :, :] = x_st[:, t, :, :, :] * mask
        else:
            x_sc_out = x_sc
            x_st_out = x_st

        return x_st_out, x_sc_out


class Linear(nn.Module):

    def __init__(self, D_in, D_out, net_params, device=torch.device('cpu'), bias=True):
        super(Linear, self).__init__()

        self.net_params = net_params
        self.linearif = LinearIF.apply
        self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
        self.device = device

    def forward(self, input_feature_st, input_features_sc):
        # weight update based on the surrogate linear layer
        T = input_feature_st.shape[1]
        output_round = torch.floor(self.linear(input_features_sc))
        output = torch.clamp(output_round, min=0, max=T)

        # extract the weight and bias from the surrogate linear layer
        linearif_weight = self.linear.weight.detach().to(self.device)
        linearif_bias = self.linear.bias.detach().to(self.device)

        # propagate the input spike train through the linearIF layer to get actual output
        # spike train
        output_st, output_sc = self.linearif(input_feature_st, output, linearif_weight, self.net_params, \
                                             self.device, linearif_bias)

        return output_st, output_sc


class LinearDropout1d(nn.Module):

    def __init__(self, D_in, D_out, device=torch.device('cuda:0'), bias=True, drop=0.5):
        super(LinearDropout1d, self).__init__()
        self.linearif = LinearIF.apply
        self.device = device
        self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
        self.dropout = nn.Dropout(drop)

    def forward(self, input_feature_st, input_features_sc):
        # weight update based on the surrogate linear layer
        T = input_feature_st.shape[1]
        output_bn = self.dropout(self.linear(input_features_sc))
        output = F.relu(output_bn)

        # extract the weight and bias from the surrogate linear layer
        linearif_weight = self.linear.weight  # .detach().to(self.device)
        linearif_bias = self.linear.bias  # .detach().to(self.device)

        # bnGamma = self.bn1d.weight
        # bnBeta = self.bn1d.bias
        # bnMean = self.bn1d.running_mean
        # bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Linear' layer weights
        # ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        # weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0)
        # biasNorm = torch.mul(linearif_bias-bnMean, ratio) + bnBeta

        # propagate the input spike train through the linearIF layer to get actual output
        # spike train
        output_st, output_sc = self.linearif(input_feature_st, output, linearif_weight, \
                                             self.device, linearif_bias)

        return output_st, output_sc


class ConvBN1d(nn.Module):

    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
                 padding=0, bias=True, weight_init=2.0, eps=1-17, momentum=0.9):
        super(ConvBN1d, self).__init__()
        self.conv1dIF = Conv1dIF.apply
        self.conv1d = torch.nn.Conv1d(Cin, Cout, kernel_size, stride, padding, bias=bias)
        self.bn1d = torch.nn.BatchNorm1d(Cout, eps=eps, momentum=momentum)
        self.device = device
        self.stride = stride
        self.padding = padding
        # nn.init.normal_(self.bn1d.weight, 0, weight_init)
        # nn.init.uniform_(self.conv1d.weight, -0.1, 0.1)
        # nn.init.uniform_(self.conv1d.bias, -0.1, 0.1)
        nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
        nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

    def forward(self, input_feature_st, input_features_sc):
        # T = input_feature_st.shape[1]

        # weight update based on the surrogate conv2d layer
        output_bn = self.bn1d(self.conv1d(input_features_sc))
        output = F.relu(output_bn)
        # output = torch.clamp(output_bn, min=0, max=T)

        # extract the weight and bias from the surrogate conv layer
        conv1d_weight = self.conv1d.weight.detach().to(self.device)
        conv1d_bias = self.conv1d.bias.detach().to(self.device)

        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(conv1d_weight.permute(1, 2, 0), ratio).permute(2, 0, 1)
        biasNorm = torch.mul(conv1d_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the IF layer to get actual output
        # spike train
        output_features_st, output_features_sc = self.conv1dIF(input_feature_st, output, \
                                                               weightNorm, self.device, \
                                                               biasNorm, self.stride, self.padding)

        return output_features_st, output_features_sc

class LinearBN1d_v4(nn.Module):

    def __init__(self, D_in, D_out, device=torch.device(SETTINGS.training.device), bias=True,last_layer=False):
        super(LinearBN1d_v4, self).__init__()
        self.linearif = LinearIF_v2.apply
        self.device = device
        self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
        # self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-4, momentum=0.9)
        # if not last_layer:
        self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-6, momentum=0.1, affine=True, track_running_stats=True)
        # self.Vthres = nn.Parameter(torch.ones(1),requires_grad=True)
        self.laset_layer = last_layer
    # nn.init.normal_(self.bn1d.weight, 0, 2.0)

    def forward(self, x, input_feature_st, input_features_sc):
        # weight update based on the surrogate linear layer
        if self.laset_layer:
            x = self.bn1d(self.linear(x))
            x_sc = self.linear(input_features_sc)
            return x, input_feature_st, x_sc
        output_bn = self.bn1d(self.linear(input_features_sc))
        output = F.relu(output_bn)

        x = F.relu(self.bn1d(self.linear(x)))
        # extract the weight and bias from the surrogate linear layer
        # linearif_weight = self.linear.weight  # .detach().to(self.device)
        # linearif_bias = self.linear.bias  # .detach().to(self.device)

        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Linear' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(self.linear.weight.permute(1, 0), ratio).permute(1, 0)
        biasNorm = torch.mul(self.linear.bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the linearIF layer to get actual output
        # spike train
        output_st, output_sc = self.linearif(input_feature_st, output, weightNorm, \
                                             self.device, biasNorm)

        return x, output_st, output_sc

class ConvBN1d_v4(nn.Module):

    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
                 padding=0, bias=True, weight_init=2.0, eps=1-17, momentum=0.9,last_layer=False):
        super(ConvBN1d_v4, self).__init__()
        self.conv1dIF = Conv1dIF.apply
        if not last_layer:
            self.bn1d = torch.nn.BatchNorm1d(Cout, eps=eps, momentum=momentum)
        self.conv1d = torch.nn.Conv1d(Cin, Cout, kernel_size, stride, padding, bias=bias)
        self.device = device
        self.stride = stride
        self.padding = padding
        self.laset_layer = last_layer
        # if last_layer:
        #     nn.init.normal_(self.bn1d.weight, 0, 2.0)
        # nn.init.uniform_(self.conv1d.weight, -0.1, 0.1)
        # nn.init.uniform_(self.conv1d.bias, -0.1, 0.1)
        nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
        nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

    def forward(self, x, input_feature_st, input_features_sc):
        # T = input_feature_st.shape[1]
        if self.laset_layer:
            x = (self.conv1d(x))#self.bn1d
            x_sc = self.conv1d(input_features_sc)
            return x, input_feature_st, x_sc

        # weight update based on the surrogate conv2d layer
        output_bn = self.bn1d(self.conv1d(input_features_sc))
        output = F.relu(output_bn)
        # output = torch.clamp(output_bn, min=0, max=T)
        x = F.relu(self.bn1d(self.conv1d(x)))

        # extract the weight and bias from the surrogate conv layer
        conv1d_weight = self.conv1d.weight.detach().to(self.device)
        conv1d_bias = self.conv1d.bias.detach().to(self.device)

        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(conv1d_weight.permute(1, 2, 0), ratio).permute(2, 0, 1)
        biasNorm = torch.mul(conv1d_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the IF layer to get actual output
        # spike train
        output_features_st, output_features_sc = self.conv1dIF(input_feature_st, output, \
                                                               weightNorm, self.device, \
                                                               biasNorm, self.stride, self.padding)

        return x, output_features_st, output_features_sc


class ConvBN1d_v5(nn.Module):

    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
                 padding=0, bias=True, weight_init=2.0, eps=1-17, momentum=0.9,last_layer=False,pooling=1):
        super(ConvBN1d_v5, self).__init__()
        self.conv1dIF = Conv1dIF.apply
        if not last_layer:
            self.bn1d = torch.nn.BatchNorm1d(Cout, eps=eps, momentum=momentum)
        self.conv1d = torch.nn.Conv1d(Cin, Cout, kernel_size, stride, padding, bias=bias)
        self.device = device
        self.stride = stride
        self.padding = padding
        self.laset_layer = last_layer
        self.pooling = pooling
        # if last_layer:
        #     nn.init.normal_(self.bn1d.weight, 0, 2.0)
        # nn.init.uniform_(self.conv1d.weight, -0.1, 0.1)
        # nn.init.uniform_(self.conv1d.bias, -0.1, 0.1)
        nn.init.uniform_(self.conv1d.weight, -0.5, 0.5)
        nn.init.uniform_(self.conv1d.bias, -0.5, 0.5)

    def forward(self, x, input_feature_st, input_features_sc):
        # T = input_feature_st.shape[1]
        if self.laset_layer:
            x = F.avg_pool1d(self.conv1d(x),self.pooling)#self.bn1d
            x_sc = F.avg_pool1d(self.conv1d(input_features_sc),self.pooling)
            return x, input_feature_st, x_sc

        # weight update based on the surrogate conv2d layer
        output_bn = F.avg_pool1d(self.bn1d(self.conv1d(input_features_sc)),self.pooling)
        output = F.relu(output_bn)
        # output = torch.clamp(output_bn, min=0, max=T)
        x = F.relu(F.avg_pool1d(self.bn1d(self.conv1d(x)),self.pooling))

        # extract the weight and bias from the surrogate conv layer
        conv1d_weight = self.conv1d.weight.detach().to(self.device)
        conv1d_bias = self.conv1d.bias.detach().to(self.device)

        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(conv1d_weight.permute(1, 2, 0), ratio).permute(2, 0, 1)
        biasNorm = torch.mul(conv1d_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the IF layer to get actual output
        # spike train
        output_features_st, output_features_sc = self.conv1dIF(input_feature_st, output, \
                                                               weightNorm, self.device, \
                                                               biasNorm, self.stride, self.padding,self.pooling)

        return x, output_features_st, output_features_sc

# class ConvBlockBN1d(nn.Module):
#     def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
#                  padding=0, bias=True, weight_init=2.0,momentum=0.1,eps=1e-4):
#         super(ConvBlockBN1d, self).__init__()
#         self.Conv1 = ConvBN1d(Cin, Cout // 2, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum)
#         self.Conv2 = ConvBN1d(Cout // 2, Cout, 3, device, 1, padding=1, bias=bias, eps=eps, momentum=momentum)
#         self.Conv3 = ConvBN1d(Cin, Cout, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum)
#
#     def forward(self, x_st, x):
#         st1, sc1 = self.Conv1(x_st, x)
#         st3, sc3 = self.Conv2(st1, sc1)
#         st2, sc2 = self.Conv3(x_st, x)
#         output_features_sc = (sc2 + sc3)/2
#         output_features_st = (st2 + st3).clamp(max=1)
#
#         return output_features_st, output_features_sc

class ConvBlockBN1d(nn.Module):
    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
                 padding=0, bias=True, weight_init=2.0,momentum=0.1,eps=1e-4):
        super(ConvBlockBN1d, self).__init__()
        self.Cout = Cout
        self.Cin = Cin
        self.kernel_size = kernel_size
        self.Conv1 = ConvBN1d(Cin, Cout // 2, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum)
        self.Conv2 = ConvBN1d(Cout // 2, Cout // 2, kernel_size, device, stride, padding=padding, bias=bias, eps=eps,
                                 momentum=momentum)
        self.Conv3 = ConvBN1d(Cin, Cout // 2, 1, device, 1, 0, bias, eps=eps, momentum=momentum)

        self.energy = 0

    def forward(self, x_st, x):
        energy = 0
        st1, sc1 = self.Conv1(x_st, x)
        energy += (st1.detach().cpu().sum((1, 2, 3)) * self.kernel_size * self.Cin).sum()
        st2, sc2 = self.Conv2(st1, sc1)
        energy += (st2.detach().cpu().sum((1, 2, 3)) * self.kernel_size* self.Cout // 2).sum()
        st3, sc3 = self.Conv3(x_st, x)
        energy += (st3.detach().cpu().sum((1, 2, 3)) * 1 * self.Cin).sum()
        output_features_sc = torch.cat((sc2, sc3),dim=1) #torch.where(sc2>sc3, sc2, sc3)#sc2 + sc3)/2
        output_features_st = torch.cat((st2, st3), dim=2)
        # output_features_sc = (sc2 + sc3) / 2
        # output_features_st = (st2 + st3).clamp(max=1)
        self.energy = energy
        return output_features_st, output_features_sc

class ConvBlockBN1d_v2(nn.Module):
    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
                 padding=0, bias=True, weight_init=2.0,momentum=0.1,eps=1e-4):
        super(ConvBlockBN1d_v2, self).__init__()
        self.Cout = Cout
        self.Cin = Cin
        self.kernel_size = kernel_size
        # self.Conv1 = ConvBN1d_v4(Cin, Cout // 2, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum)
        # self.Conv2 = ConvBN1d_v4(Cout // 2, Cout//2, 3, device, 1, padding=1, bias=bias, eps=eps, momentum=momentum)
        # self.Conv3 = ConvBN1d_v4(Cin, Cout//2, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum)
        self.Conv1 = ConvBN1d_v4(Cin, Cout//2, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum)
        self.Conv2 = ConvBN1d_v4(Cout // 2, Cout//2, kernel_size, device, stride, padding=padding, bias=bias, eps=eps, momentum=momentum)
        self.Conv3 = ConvBN1d_v4(Cin, Cout//2, 1, device, 1, 0, bias, eps=eps, momentum=momentum)

    def forward(self, x, x_st, x_sc):
        energy = 0
        x1,st1, sc1 = self.Conv1(x, x_st, x_sc)
        energy += (st1.detach().cpu().sum((1, 2, 3)) * self.kernel_size * self.Cin).sum()
        x2, st2, sc2 = self.Conv2(x1,st1, sc1)
        energy += (st2.detach().cpu().sum((1, 2, 3)) * self.kernel_size * self.Cout // 2).sum()
        x3, st3, sc3 = self.Conv3(x, x_st, x_sc)
        energy += (st3.detach().cpu().sum((1, 2, 3)) * 1 * self.Cin).sum()
        output_features_x = torch.cat((x2, x3), dim=1)
        output_features_sc = torch.cat((sc2, sc3),dim=1) #torch.where(sc2>sc3, sc2, sc3)#sc2 + sc3)/2
        output_features_st = torch.cat((st2, st3), dim=2)
        # output_features_sc = (sc2 + sc3) / 2
        # output_features_st = (st2 + st3).clamp(max=1)
        self.energy = energy
        return output_features_x, output_features_st, output_features_sc


class ConvBlockBN1d_v3(nn.Module):
    def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, \
                 padding=0, bias=True, weight_init=2.0,momentum=0.1,eps=1e-4):
        super(ConvBlockBN1d_v3, self).__init__()
        self.Cout = Cout
        self.Cin = Cin
        self.kernel_size = kernel_size
        # self.Conv1 = ConvBN1d_v4(Cin, Cout // 2, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum)
        # self.Conv2 = ConvBN1d_v4(Cout // 2, Cout//2, 3, device, 1, padding=1, bias=bias, eps=eps, momentum=momentum)
        # self.Conv3 = ConvBN1d_v4(Cin, Cout//2, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum)
        self.Conv1 = ConvBN1d_v5(Cin, Cout//2, kernel_size, device, stride, padding, bias, eps=eps, momentum=momentum,pooling=2)
        self.Conv2 = ConvTransposeBN1d_v2(Cout // 2, Cout//2, 4, device, 2, padding=1, bias=bias, eps=eps, momentum=momentum)
        self.Conv3 = ConvBN1d_v5(Cin, Cout//2, 1, device, 1, 0, bias, eps=eps, momentum=momentum)

    def forward(self, x, x_st, x_sc):
        # energy = 0
        x1,st1, sc1 = self.Conv1(x, x_st, x_sc)

        # energy += (st1.detach().cpu().sum((1, 2, 3)) * self.kernel_size * self.Cin).sum()
        x2, st2, sc2 = self.Conv2(x1,st1, sc1)
        # energy += (st3.detach().cpu().sum((1, 2, 3)) * 3 * self.Cout // 2).sum()
        x3, st3, sc3 = self.Conv3(x, x_st, x_sc)
        # energy += (st2.detach().cpu().sum((1, 2, 3)) * self.kernel_size * self.Cin).sum()
        output_features_x = torch.cat((x2, x3), dim=1)

        output_features_sc = torch.cat((sc2, sc3),dim=1) #torch.where(sc2>sc3, sc2, sc3)#sc2 + sc3)/2
        output_features_st = torch.cat((st2, st3), dim=2)
        # output_features_sc = (sc2 + sc3) / 2
        # output_features_st = (st2 + st3).clamp(max=1)
        # self.energy = energy
        return output_features_x, output_features_st, output_features_sc


class actF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pot_aggregate, spike_mask):
        """
		args:

		"""

        spike = pot_aggregate.ge(1.0).float()
        spike *= (1 - spike_mask)
        ctx.save_for_backward(spike)
        # print(feature_out.requires_grad)

        return spike

    @staticmethod
    def backward(ctx, grad_feature_out):
        spike, = ctx.saved_tensors
        grad_feature_in = grad_feature_out.clone()

        return grad_feature_in * (-spike), None


actF = actF.apply


class ConvBN2d_(nn.Module):
    """
	W
	"""

    def __init__(self, Cin, Cout, kernel_size, device=torch.device('cuda:2'), stride=1, padding=0, bias=True,
                 weight_init=2.0, pooling=1):
        super(ConvBN2d_, self).__init__()
        self.conv2dIF = Conv2dIF.apply
        self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
        # self.bn2d = torch.nn.BatchNorm2d(Cout, eps=1e-4, momentum=0.9)
        self.bn2d = torch.nn.BatchNorm2d(Cout)
        self.device = device
        self.stride = stride
        self.padding = padding
        self.pooling = pooling
        nn.init.normal_(self.bn2d.weight, 0, weight_init)

    def forward(self, input_feature_st, input_features_sc):
        # weight update based on the surrogate conv2d layer
        output_bn = F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)), self.pooling)
        ann_output = F.relu(output_bn)
        # output = torch.clamp(output_bn, min=0, max=T)

        # extract the weight and bias from the surrogate conv layer
        conv2d_weight = self.conv2d.weight.detach().to(self.device)
        conv2d_bias = self.conv2d.bias.detach().to(self.device)

        bnGamma = self.bn2d.weight
        bnBeta = self.bn2d.bias
        bnMean = self.bn2d.running_mean
        bnVar = self.bn2d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(conv2d_weight.permute(1, 2, 3, 0), ratio).permute(3, 0, 1, 2)
        biasNorm = torch.mul(conv2d_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the IF layer to get actual output
        # spike train
        # output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output, \
        # 													   weightNorm, self.device, biasNorm, \
        # 													   self.stride, self.padding, self.pooling)
        N, T, in_channels, iH, iW = input_feature_st.shape
        out_channels, in_channels, kH, kW = weightNorm.shape
        pot_aggregate = torch.zeros_like(output_bn)  # init the membrane potential with the bias
        _, _, outH, outW = pot_aggregate.shape
        spike_out = torch.zeros(N, T, out_channels, outH, outW, device=self.device)
        spike_mask = torch.zeros_like(pot_aggregate, device=self.device).float()
        spike_count_out = torch.zeros_like(spike_out[:, 0, :, :, :])

        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate += F.max_pool2d(
                F.conv2d(input_feature_st[:, t, :, :, :], weightNorm, biasNorm, self.stride, self.padding),
                self.pooling)
            bool_spike = actF(pot_aggregate, spike_mask)

            spike_count_out += bool_spike
            spike_out[:, t, :, :, :] = bool_spike
            pot_aggregate -= bool_spike

            spike_mask += bool_spike
            spike_mask[spike_mask > 0] = 1

        return spike_out, spike_count_out, ann_output






class LinearBN1d_(nn.Module):

    def __init__(self, D_in, D_out, device=torch.device('cuda:2'), bias=True):
        super(LinearBN1d_, self).__init__()
        self.linearif = LinearIF.apply
        self.device = device
        self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
        # self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-4, momentum=0.9)
        self.bn1d = torch.nn.BatchNorm1d(D_out)
        nn.init.normal_(self.bn1d.weight, 0, 2.0)

    def forward(self, input_feature_st, input_features_sc):
        # weight update based on the surrogate linear layer
        T = input_feature_st.shape[1]
        output_bn = self.bn1d(self.linear(input_features_sc))
        ann_output = F.relu(output_bn)

        # extract the weight and bias from the surrogate linear layer
        linearif_weight = self.linear.weight  # .detach().to(self.device)
        linearif_bias = self.linear.bias  # .detach().to(self.device)

        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Linear' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0)
        biasNorm = torch.mul(linearif_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the linearIF layer to get actual output
        # spike train
        # output_st, output_sc = self.linearif(input_feature_st, output, weightNorm,  \
        # 										self.device, biasNorm)
        N, T, _ = input_feature_st.shape
        pot_in = input_feature_st.matmul(weightNorm.t())
        spike_out = torch.zeros_like(pot_in, device=self.device)
        pot_aggregate = biasNorm.repeat(N, 1)  # init the membrane potential with the bias
        spike_mask = torch.zeros_like(pot_aggregate, device=self.device).float()
        spike_count_out = torch.zeros_like(spike_out[:, 0, :])

        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate += pot_in[:, t, :].squeeze()
            bool_spike = actF(pot_aggregate, spike_mask)
            spike_count_out += bool_spike
            spike_out[:, t, :] = bool_spike
            pot_aggregate -= bool_spike

            spike_mask += bool_spike
            spike_mask[spike_mask > 0] = 1

        return spike_out, spike_count_out, ann_output
