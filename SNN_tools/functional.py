# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:49:34 2020

@author: win10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
from settings import SETTINGS
# print(os.path.dirname(os.getcwd()))
# print(os.path.dirname(os.getcwd()))
class ConvTranspose2dIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spike_in, features_in, weight, device=torch.device(SETTINGS.training.device), bias=None, stride=1, padding=0,output_padding=0, pooling=1):
        """
        args:
            spike_in: (N, T, in_channels, iH, iW)
            features_in: (N, in_channels, iH, iW)
            weight: (out_channels, in_channels, kH, kW)
            bias: (out_channels)
        """
        N, T, in_channels, iH, iW = spike_in.shape
        in_channels, out_channels,kH, kW = weight.shape
        pot_aggregate = F.max_pool2d(F.conv_transpose2d(torch.zeros_like(spike_in[:,0,:,:,:]), weight, bias, stride, padding,output_padding), pooling) # init the membrane potential with the bias
        _, _, outH, outW = pot_aggregate.shape
        spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
        # spike_mask = torch.zeros_like(pot_aggregate,device=device).float()


        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate += F.max_pool2d(F.conv_transpose2d(spike_in[:,t,:,:,:], weight, None, stride, padding,output_padding), pooling)
            bool_spike = torch.ge(pot_aggregate, 1.0).float()

            # bool_spike *=(1-spike_mask)
            spike_out[:,t,:,:,:] = bool_spike
            pot_aggregate -= bool_spike

            # spike_mask += bool_spike
            # spike_mask[spike_mask > 0] = 1

        spike_count_out = torch.sum(spike_out, dim=1)

        # spike_count_out = (spike_out * T_window).sum(dim=1)
        # spike_count_out = T - spike_count_out
        # print((spike_count_out>1).float().sum())
        return spike_out, spike_count_out
    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        grad_spike_count_out = grad_spike_count_out.clone()

        grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding, grad_output_padding,grad_pooling = None, \
                None, None, None, None, None, None, None

        return grad_spike_in, grad_spike_count_out, grad_weight, grad_device, grad_bias, \
                grad_stride, grad_padding, grad_output_padding,grad_pooling


class ConvTranspose1dIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spike_in, features_in, weight, device=torch.device(SETTINGS.training.device), bias=None, stride=1, padding=0,output_padding=0, pooling=1):
        """
        args:
            spike_in: (N, T, in_channels, iH, iW)
            features_in: (N, in_channels, iH, iW)
            weight: (out_channels, in_channels, kH, kW)
            bias: (out_channels)
        """
        N, T, in_channels, iH = spike_in.shape
        in_channels, out_channels,kH = weight.shape
        pot_aggregate = F.avg_pool1d(F.conv_transpose1d(torch.zeros_like(spike_in[:,0,:,:]), weight, bias, stride, padding,output_padding), pooling) # init the membrane potential with the bias
        bias_distribute = F.conv_transpose1d(torch.zeros_like(spike_in[:, 0, :, :]), weight, bias, stride, padding,output_padding)/ T
        _, _, outH = pot_aggregate.shape
        spike_out = torch.zeros(N, T, out_channels, outH,  device=device)
        # spike_mask = torch.zeros_like(pot_aggregate,device=device).float()


        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate += F.avg_pool1d(F.conv_transpose1d(spike_in[:,t,:,:,], weight, None, stride, padding,output_padding)+bias_distribute, pooling)
            bool_spike = torch.ge(pot_aggregate, 1.0).float()

            # bool_spike *=(1-spike_mask)
            spike_out[:,t,:,:,] = bool_spike
            pot_aggregate -= bool_spike

            # spike_mask += bool_spike
            # spike_mask[spike_mask > 0] = 1

        spike_count_out = torch.sum(spike_out, dim=1)

        # spike_count_out = (spike_out * T_window).sum(dim=1)
        # spike_count_out = T - spike_count_out
        # print((spike_count_out>1).float().sum())
        return spike_out, spike_count_out
    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        grad_spike_count_out = grad_spike_count_out.clone()

        grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding, grad_output_padding,grad_pooling = None, \
                None, None, None, None, None, None, None

        return grad_spike_in, grad_spike_count_out, grad_weight, grad_device, grad_bias, \
                grad_stride, grad_padding, grad_output_padding,grad_pooling

class LinearIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spike_in, ann_output, weight, device=torch.device(SETTINGS.training.device), bias=None):
        """
        args:
            spike_in: (N, T, in_features)
            weight: (out_features, in_features)
            bias: (out_features)
        """
        N, T, _ = spike_in.shape
        out_features = bias.shape[0]
        pot_in = spike_in.matmul(weight.t())
        spike_out = torch.zeros_like(pot_in, device=device)
        pot_aggregate = bias.repeat(N, 1) # init the membrane potential with the bias
        # spike_mask = torch.zeros_like (pot_aggregate, device=device).float ()


        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate += pot_in[:,t,:].squeeze()
            bool_spike = torch.ge(pot_aggregate, 1.0).float()

            # bool_spike *= (1 - spike_mask)

            spike_out[:,t,:] = bool_spike
            pot_aggregate -= bool_spike

            # spike_mask += bool_spike
            # spike_mask[spike_mask > 0] = 1

        spike_count_out = torch.sum(spike_out, dim=1).squeeze()


        return spike_out, spike_count_out

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""
        grad_ann_out = grad_spike_count_out.clone()

        return None, grad_ann_out, None, None, None, None
class LinearIF_v2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spike_in, ann_output, weight,  device=torch.device(SETTINGS.training.device), bias=None):
        """
        args:
            spike_in: (N, T, in_features)
            weight: (out_features, in_features)
            bias: (out_features)
        """
        N, T, _ = spike_in.shape
        # out_features = bias.shape[0]
        pot_in = spike_in.matmul(weight.t())
        spike_out = torch.zeros_like(pot_in,device=device)
        bias_distribute = bias.repeat(N, 1) / T
        pot_aggregate = 0#torch.zeros(N, out_features,device=device)

        # pot_aggregate = bias.repeat(N, 1) # init the membrane potential with the bias
        # spike_mask = torch.zeros_like (pot_aggregate, device=device).float ()

        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate = pot_aggregate+(pot_in[:, t, :].squeeze() + bias_distribute)
            bool_spike = torch.ge(pot_aggregate, 1).float()

            # bool_spike *= (1 - spike_mask)

            spike_out[:,t,:] = bool_spike
            pot_aggregate = pot_aggregate - bool_spike

            # spike_mask += bool_spike
            # spike_mask[spike_mask > 0] = 1
        spike_count_out = torch.sum(spike_out, dim=1).squeeze()
        # ctx.save_for_backward(ann_output,spike_count_out)
        return spike_out, spike_count_out

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""
        grad_ann_out = grad_spike_count_out.clone()

        return None, grad_ann_out, None, None, None, None
class Conv2dIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spike_in, features_in, weight, device=torch.device(SETTINGS.training.device), bias=None, stride=1, padding=0, pooling=1):
        """
        args:
            spike_in: (N, T, in_channels, iH, iW)
            features_in: (N, in_channels, iH, iW)
            weight: (out_channels, in_channels, kH, kW)
            bias: (out_channels)
        """
        N, T, in_channels, iH, iW = spike_in.shape
        out_channels, in_channels, kH, kW = weight.shape
        pot_aggregate = F.max_pool2d(F.conv2d(torch.zeros_like(spike_in[:,0,:,:,:]), weight, bias, stride, padding), pooling) # init the membrane potential with the bias
        _, _, outH, outW = pot_aggregate.shape
        spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
        # spike_mask = torch.zeros_like(pot_aggregate,device=device).float()


        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate += F.max_pool2d(F.conv2d(spike_in[:,t,:,:,:], weight, None, stride, padding), pooling)
            bool_spike = torch.ge(pot_aggregate, 1.0).float()

            # bool_spike *=(1-spike_mask)

            spike_out[:,t,:,:,:] = bool_spike
            pot_aggregate -= bool_spike

            # spike_mask += bool_spike
            # spike_mask[spike_mask > 0] = 1

        spike_count_out = torch.sum(spike_out, dim=1)

        # spike_count_out = (spike_out * T_window).sum(dim=1)
        # spike_count_out = T - spike_count_out
        # print((spike_count_out>1).float().sum())
        return spike_out, spike_count_out
    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        grad_spike_count_out = grad_spike_count_out.clone()

        grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding, grad_pooling = None, \
                None, None, None, None, None, None

        return grad_spike_in, grad_spike_count_out, grad_weight, grad_device, grad_bias, \
                grad_stride, grad_padding, grad_pooling

class Conv2dIF_v2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spike_in, features_in, weight, device=torch.device(SETTINGS.training.device), bias=None, stride=1, padding=0, pooling=1):
        """
        args:
            spike_in: (N, T, in_channels, iH, iW)
            features_in: (N, in_channels, iH, iW)
            weight: (out_channels, in_channels, kH, kW)
            bias: (out_channels)
        """
        N, T, in_channels, iH, iW = spike_in.shape
        out_channels, in_channels, kH, kW = weight.shape
        # pot_aggregate = F.max_pool2d(F.conv2d(torch.zeros_like(spike_in[:,0,:,:,:]), weight, bias, stride, padding), pooling) # init the membrane potential with the bias
        pot_aggregate =0#torch.zeros_like(features_in,device=device)
        _, _, outH, outW = features_in.shape
        bias_distribute = F.conv2d(torch.zeros_like(spike_in[:, 0, :, :, :]), weight, bias, stride, padding) / T
        spike_out = torch.zeros(N, T, out_channels, outH, outW,device=device)
        # spike_mask = torch.zeros_like(pot_aggregate,device=device).float()


        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate = pot_aggregate + F.max_pool2d(F.conv2d(spike_in[:,t,:,:,:], weight, None, stride, padding)+bias_distribute, pooling)
            bool_spike = torch.ge(pot_aggregate, 1.0).float()

            # bool_spike *=(1-spike_mask)

            spike_out[:,t,:,:,:] = bool_spike
            pot_aggregate = pot_aggregate - bool_spike

            # spike_mask += bool_spike
            # spike_mask[spike_mask > 0] = 1

        spike_count_out = torch.sum(spike_out, dim=1)

        # spike_count_out = (spike_out * T_window).sum(dim=1)
        # spike_count_out = T - spike_count_out
        # print((spike_count_out>1).float().sum())
        return spike_out, spike_count_out

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        grad_spike_count_out = grad_spike_count_out.clone()


        return None, grad_spike_count_out, None, None, None, \
                None, None, None







class ZeroExpandInput_CNN_DoA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_image, T, device=torch.device(SETTINGS.training.device)):
        """
        Args:
            input_image: normalized within (0,1)
        """
        if len(input_image.shape)==4:
            batch_size,channel, spec_length, fing_width = input_image.shape
            zero_inputs = torch.zeros(batch_size, T - 1, channel, spec_length, fing_width).to(device)
        else:
            batch_size, channel, spec_length= input_image.shape
            zero_inputs = torch.zeros(batch_size, T - 1, channel, spec_length).to(device)
        input_image_sc = input_image
        input_image = input_image.unsqueeze(dim=1)
        input_image_spike = torch.cat((input_image, zero_inputs), dim=1)
        return input_image_spike, input_image_sc

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""
        grad_spike_count_out = grad_spike_count_out.clone()

        return grad_spike_count_out, None, None

class ZeroExpandInput_CNN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_image, T, device=torch.device(SETTINGS.training.device)):
        """
        Args:
            input_image: normalized within (0,1)
        """
        #N, dim = input_image.shape
        #input_image_sc = input_image
        #zero_inputs = torch.zeros(N, T-1, dim).to(device)
        #input_image = input_image.unsqueeze(dim=1)
        #input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

        #return input_image_spike, input_image_sc
        # if len(input_image.shape)==4:
        # 	input_image = input_image.squeeze(1)
        # print(input_image.shape)
        # input_image = input_image.sum(1)
        if len(input_image.shape)==4:
            batch_size,channel, spec_length, fing_width = input_image.shape
            zero_inputs = torch.zeros(batch_size, T - 1, channel, spec_length, fing_width).to(device)
        else:
            batch_size, channel, spec_length= input_image.shape
            zero_inputs = torch.zeros(batch_size, T - 1, channel, spec_length).to(device)
        ##################################
        # input_image_tmp = (input_image-input_image.min())/(input_image.max()-input_image.min()+1e-10) # normalized to [0-1]
        # encode_window = int(T/3)-1
        # input_image_spike = torch.zeros(batch_size,T,channel,spec_length,fing_width).to(device)
        # input_sc_index =((1-input_image_tmp)*encode_window).ceil().unsqueeze(1).long()
        # input_image_spike=input_image_spike.scatter(1,input_sc_index,1).float()
        ####################################
        input_image_sc = input_image
        input_image = input_image.unsqueeze(dim=1)
        # print(input_image.device)
        # print(zero_inputs.device)
        input_image_spike = torch.cat((input_image, zero_inputs), dim=1)
        #
        return input_image_spike, input_image_sc

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        return None, None, None

class ZeroExpandInput_CNN_v2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_image, T, device=torch.device(SETTINGS.training.device)):
        """
        Args:
            input_image: normalized within (0,1)
        """
        #N, dim = input_image.shape
        #input_image_sc = input_image
        #zero_inputs = torch.zeros(N, T-1, dim).to(device)
        #input_image = input_image.unsqueeze(dim=1)
        #input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

        #return input_image_spike, input_image_sc
        # if len(input_image.shape)==4:
        # 	input_image = input_image.squeeze(1)
        # print(input_image.shape)
        # input_image = input_image.sum(1)
        x = input_image
        if len(input_image.shape)==4:
            batch_size,channel, spec_length, fing_width = input_image.shape
            zero_inputs = torch.zeros(batch_size, T - 1, channel, spec_length, fing_width).to(device)
        else:
            batch_size, channel, spec_length= input_image.shape
            zero_inputs = torch.zeros(batch_size, T - 1, channel, spec_length).to(device)
        ##################################
        # input_image_tmp = (input_image-input_image.min())/(input_image.max()-input_image.min()+1e-10) # normalized to [0-1]
        # encode_window = int(T/3)-1
        # input_image_spike = torch.zeros(batch_size,T,channel,spec_length,fing_width).to(device)
        # input_sc_index =((1-input_image_tmp)*encode_window).ceil().unsqueeze(1).long()
        # input_image_spike=input_image_spike.scatter(1,input_sc_index,1).float()
        ####################################
        input_image_sc = input_image
        input_image = input_image.unsqueeze(dim=1)
        input_image_spike = torch.cat((input_image, zero_inputs), dim=1)
        #
        return x, input_image_spike, input_image_sc

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        return None, None, None

class ZeroExpandInput_CNN_Sin(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_image, T, device=torch.device('cuda:0')):
        """
        Args:
            input_image: normalized within (0,1)
        """
        #N, dim = input_image.shape
        #input_image_sc = input_image
        #zero_inputs = torch.zeros(N, T-1, dim).to(device)
        #input_image = input_image.unsqueeze(dim=1)
        #input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

        #return input_image_spike, input_image_sc
        if len(input_image.shape)==4:
            input_image = input_image.squeeze(1)
        batch_size, spec_length, fing_width = input_image.shape

        input_image = (input_image.clamp(min=-247,max=30)+247)/277
        # input_image = torch.sin(input_image*math.pi/2)
        # input_image = torch.sin(input_image*math.pi/2)
        input_image_sc = input_image
        zero_inputs = torch.zeros(batch_size, T-1, spec_length, fing_width).to(device)
        input_image = input_image.unsqueeze(dim=1)
        input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

        return input_image_spike, input_image_sc

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        return None, None, None

class ZeroExpandInput_MLP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_image, T, device=torch.device(SETTINGS.training.device)):
        """
        Args:
            input_image: normalized within (0,1)
        """

        N, dim = input_image.shape
        input_image_sc = input_image
        zero_inputs = torch.zeros(N, T-1, dim).to(device)
        input_image = input_image.unsqueeze(dim=1)
        input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

        return input_image_spike, input_image_sc

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        return None, None, None

class ZeroExpandInput_CNN_Spike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_image, Tencode=1, device=torch.device('cuda:0')):
        """
        Args:
            input_image: normalized within (0,1)
        """
        #N, dim = input_image.shape
        #input_image_sc = input_image
        #zero_inputs = torch.zeros(N, T-1, dim).to(device)
        #input_image = input_image.unsqueeze(dim=1)
        #input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

        #return input_image_spike, input_image_sc
        if len(input_image.shape)==4:
            input_image = input_image.squeeze(1)
        batch_size, spec_length, fing_width = input_image.shape


        # input_image = torch.sin(input_image*math.pi/2)
        # input_image = torch.sin(input_image*math.pi/2)
        input_image_sc = input_image
        # zero_inputs = torch.zeros(batch_size, T-1, spec_length, fing_width).to(device)
        # input_image = input_image.unsqueeze(dim=1)
        input_image_spike = (1-input_image)*Tencode#torch.cat((input_image, zero_inputs), dim=1)

        return input_image_spike, input_image_sc

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        return None, None, None

class LinearPoissonInput(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_image, net_params, device=torch.device('cuda')):
        """
        Args:
            input_image: normalized within (0,1)
            net_params: max_fire_rate, dt, T
        """
        rescale_factor = 1/(net_params['max_fire_rate']*net_params['dt'])
        input_image_rep = input_image.unsqueeze(dim=1).expand(-1, net_params['T'], -1)# init the membrane potential with the bias
        rand_input = torch.rand_like(input_image_rep, device=device, dtype=torch.float32)*rescale_factor
        input_image_spike = torch.lt(rand_input, input_image_rep).float()
        input_image_spike_count = torch.sum(input_image_spike, dim=1).squeeze()

        return input_image_spike, input_image_spike_count

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""
        grad_input, grad_net, grad_device = None, None, None

        return grad_input, grad_net, grad_device

class LinearImage(torch.autograd.Function):

    @staticmethod
    def forward(ctx, layerIdx, features_in, net_params, device=torch.device('cuda')):
        """
        args:
            features_in: (N, in_features) inputs normalized within [0,1]
            spike_out: (N, T, in_features)
        """
        N, in_features = features_in.shape
        T = net_params["T"]
        pot_aggregate = features_in * T
        spike_out = torch.zeros(N, T, in_features, device=device)

        # Iterate over simulation time steps to determine output spike trains
        for t in range(layerIdx, T):
            bool_spike = torch.gt(pot_aggregate, net_params["Vthr"]).float()
            spike_out[:,t,:] = bool_spike
            pot_aggregate -= bool_spike * net_params["Vthr"]

        return spike_out

    @staticmethod
    def backward(ctx, grad_spike_out):
        """Auxiliary function only, no gradient required"""
        grad_layer_idx, grad_features_in, grad_net, grad_device = None, \
                None, None, None

        return grad_layer_idx, grad_features_in, grad_net, grad_device

class LinearImageInput(nn.Module):

    def __init__(self, D_in, D_out, net_params, layerIdx=0, device=torch.device('cuda'), bias=False):
        super(LinearImageInput, self).__init__()

        self.net_params = net_params
        self.linearImageIF = LinearImage.apply
        self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
        nn.init.uniform_(self.linear.weight, 0, self.net_params['T']*2.0/float(D_in))
        self.device = device
        self.layerIdx = layerIdx

    def forward(self, input_features):
        # extract the weight and bias from the surrogate linear layer
        linearif_weight = self.linear.weight.detach()

        # determine the output spike train
        output_features_spikes = self.linearImageIF(self.layerIdx, input_features, linearif_weight, \
                                                    self.net_params, self.device)

        # weight update based on the surrogate linear layer
        output_feature = self.linear(input_features)

        return output_features_spikes, output_feature


# class Conv2dIF(torch.autograd.Function):
# 	"""2D Convolutional Layer"""
#
# 	@staticmethod
# 	def forward(ctx, spike_in, features_in, weight, device=torch.device('cuda'), bias=None, stride=1, padding=0, vthr=1.0, neuronParam=None):
# 		"""
# 		Params:
# 			spike_in: input spike trains
# 			features_in: placeholder
# 			weight: connection weights
# 			device: cpu or cuda
# 			bias: neuronal bias parameters
# 			stride: stride of 1D Conv
# 			padding: padding of 1D Conv
# 			dilation: dilation of 1D Conv
# 			neuronParam： neuronal parameters
# 		Returns:
# 			spike_out: output spike trains
# 			spike_count_out: output spike counts
# 		"""
# 		supported_neuron = {
# 			'IF': IF,
# 		}
# 		if neuronParam['neuronType'] not in supported_neuron:
# 			raise RuntimeError("Unsupported Neuron Model: {}".format(neuronParam['neuronType']))
# 		N, T, in_channels, iH, iW = spike_in.shape
# 		out_channels, in_channels, kH, kW = weight.shape
# 		mem = torch.zeros_like(F.conv2d(spike_in[:, 0, :, :, :], weight, bias, stride, padding))
# 		bias_distribute = F.conv2d(torch.zeros_like(spike_in[:, 0, :, :, :]), weight, bias, stride, padding) / T
# 		_, _, outH, outW = mem.shape
# 		spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
# 		spike = torch.zeros(N, out_channels, outH, outW, device=device)  # init input spike train
#
# 		# Iterate over simulation time steps to determine output spike trains
# 		for t in range(T):
# 			x = F.conv2d(spike_in[:, t, :, :, :], weight, None, stride, padding) + bias_distribute
# 			# Membrane potential update
# 			mem, spike = IF(x, mem, spike, vthr)
# 			spike_out[:, t, :, :] = spike
#
# 		spike_count_out = torch.sum(spike_out, dim=1)
#
# 		return spike_out, spike_count_out
#
# 	@staticmethod
# 	def backward(ctx, grad_spike_out, grad_spike_count_out):
# 		"""Auxiliary function only, no gradient required"""
#
# 		grad_spike_count_out = grad_spike_count_out.clone()
#
# 		return None, grad_spike_count_out, None, None, None, None, None, None, None


class Conv1dIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spike_in, features_in, weight, device=torch.device('cuda'), bias=None,\
                    stride=1, padding=0,pooling=1):
        """
        args:
            spike_in: (N, T, in_channels, iW)
            features_in: (N, in_channels, kW)
            weight: (out_channels, in_channels, kW)
            bias: (out_channels)
        """
        N, T, in_channels, iW = spike_in.shape
        out_channels, in_channels, kW = weight.shape

        pot_aggregate = F.avg_pool1d(F.conv1d(torch.zeros_like(spike_in[:,0,:,:]), weight, bias, stride, padding),pooling) # init the membrane potential with the bias
        bias_distribute = F.conv1d(torch.zeros_like(spike_in[:, 0, :, :]), weight, bias, stride, padding) / T
        _, _, outW = pot_aggregate.shape
        # spike_mask = torch.zeros_like (pot_aggregate, device=device).float ()
        spike_out = torch.zeros(N, T, out_channels, outW, device=device)
        # T_window=torch.tensor(range(T),device=device).float().unsqueeze(1).unsqueeze(2)
        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate += F.avg_pool1d(F.conv1d(spike_in[:,t,:,:], weight, None, stride, padding)+ bias_distribute,pooling)
            bool_spike = torch.ge(pot_aggregate, 1.0).float()

            # bool_spike *= (1 - spike_mask)

            spike_out[:,t,:,:] = bool_spike
            pot_aggregate -= bool_spike

            # spike_mask += bool_spike
            # spike_mask[spike_mask > 0] = 1

        spike_count_out = torch.sum(spike_out, dim=1)
        # spike_count_out = (spike_out*T_window).sum(dim=1)/(T-1)
        # spike_count_out = 1-spike_count_out
        return spike_out, spike_count_out

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        grad_spike_count_out = grad_spike_count_out.clone()
        grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding = None, \
                None, None, None, None, None

        return grad_spike_in, grad_spike_count_out, grad_weight, grad_device, grad_bias, \
                grad_stride, grad_padding,None

class AvgPool2d_IF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, layerIdx, spike_in, features_in, kernel_size, stride=None, padding=0, device=torch.device('cuda')):
        """
        args:
            spike_in: (N, T, in_channels, iH, iW)
        """

        N, T, in_channels, iH, iW = spike_in.shape
        out_channels = in_channels

        # Iterate over simulation time steps to determine output spike trains
        for t in range(layerIdx, T):
            if t == layerIdx:
                # init the membrane potential
                pot_aggregate = F.avg_pool2d(spike_in[:,t-1,:,:,:], kernel_size, stride, padding)
                _, _, outH, outW = pot_aggregate.shape
                spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
            else:
                pot_aggregate += F.avg_pool2d(spike_in[:,t-1,:,:,:], kernel_size, stride, padding)

            bool_spike = torch.gt(pot_aggregate, 1.0).float()
            spike_out[:,t,:,:,:] = bool_spike
            pot_aggregate -= bool_spike

        spike_count_out = torch.sum(spike_in, dim=1)

        return spike_out, spike_count_out

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        grad_spike_count_out = grad_spike_count_out.clone()
        grad_layer_idx, grad_spike_in, grad_net, grad_kernel_size, grad_stride, grad_pad, grad_device = None, \
            None, None, None, None, None, None

        return grad_layer_idx, grad_spike_in, grad_spike_count_out, grad_net, grad_kernel_size, grad_stride, grad_pad, grad_device

class activate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features_in, Tmax):
        """
        args:

        """
        feature_out = torch.floor(torch.clamp(features_in, min=0, max=Tmax))
        mask_gt = features_in.gt(Tmax).float()
        mask_lt = features_in.lt(0).float()
        mask = 1-mask_gt-mask_lt
        ctx.save_for_backward(mask)
        # print(feature_out.requires_grad)

        return feature_out

    @staticmethod
    def backward(ctx, grad_feature_out):
        mask, = ctx.saved_tensors
        grad_feature_in = grad_feature_out.clone()

        return grad_feature_in*mask, None


