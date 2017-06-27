from __future__ import absolute_import
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.init_weights import init_weights, normalized_columns_initializer

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # logging
        self.logger = args.logger
        # params
        self.hidden_dim = args.hidden_dim
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        self.input_dims     = args.hist_len
        self.output_dims    = args.action_dim

    def _init_weights(self):
        raise NotImplementedError("not implemented in base class")

    def print_model(self):
        self.logger.warning("<-----------------------------------> Model")
        self.logger.warning(self)

    def _reset(self):           # NOTE: should be called at each child's __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        self.print_model()

    def forward(self, input):
        raise NotImplementedError("not implemented in base class")

class A3CCuriosity(Model):

    def __init__(self, args):
        super(A3CCuriosity, self).__init__(args)
        # Policy network:
        self.conv1 = nn.Conv2d(self.input_dims, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, self.hidden_dim)

        self.critic_linear = nn.Linear(self.hidden_dim, 1)
        self.actor_linear = nn.Linear(self.hidden_dim, self.output_dims)
        self.softmax_out = nn.Softmax()

        #############################################################
        # ICM Module
        self.icm_conv1 = nn.Conv2d(self.input_dims, 32, 3, stride=2, padding=1)
        self.icm_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.inverse_FC1 = nn.Linear(32 * 3 * 3 + 32 * 3 * 3, self.hidden_dim)
        self.inverse_FC2 = nn.Linear(self.hidden_dim, self.output_dims)

        self.forward_FC1 = nn.Linear(32 * 3 * 3 + self.output_dims, self.hidden_dim)
        self.forward_FC2 = nn.Linear(self.hidden_dim, 32 * 3 * 3)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)

        self.inverse_FC1.weight.data = normalized_columns_initializer(self.inverse_FC1.weight.data, 0.01)
        self.inverse_FC1.bias.data.fill_(0)
        self.inverse_FC2.weight.data = normalized_columns_initializer(self.inverse_FC2.weight.data, 1.0)
        self.inverse_FC2.bias.data.fill_(0)

        self.forward_FC1.weight.data = normalized_columns_initializer(self.forward_FC1.weight.data, 0.01)
        self.forward_FC1.bias.data.fill_(0)
        self.forward_FC2.weight.data = normalized_columns_initializer(self.forward_FC2.weight.data, 1.0)
        self.forward_FC2.bias.data.fill_(0)

        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, inputs, icm):

        if icm == False:
            """A3C"""
            inputs, (hx, cx) = inputs

            #x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])

            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            x = x.view(-1, 32 * 3 * 3)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx

            critic = self.critic_linear(x)
            actor = self.actor_linear(x)
            return self.softmax_out(actor), critic, (hx, cx)

        else:
            """icm"""
            s_t, s_t1, a_t = inputs

            feature_st = F.elu(self.icm_conv1(s_t))
            feature_st = F.elu(self.icm_conv2(feature_st))
            feature_st = F.elu(self.icm_conv3(feature_st))
            feature_st = F.elu(self.icm_conv4(feature_st))

            feature_st1 = F.elu(self.icm_conv1(s_t1))
            feature_st1 = F.elu(self.icm_conv2(feature_st1))
            feature_st1 = F.elu(self.icm_conv3(feature_st1))
            feature_st1 = F.elu(self.icm_conv4(feature_st1))

            feature_st = feature_st.view(-1, 32 * 3 * 3)
            feature_st1 = feature_st1.view(-1, 32 * 3 * 3)

            inverse_input = torch.cat((feature_st, feature_st1), 1)
            forward_input = torch.cat((feature_st, a_t), 1)

            inverse = self.inverse_FC1(inverse_input)
            inverse = F.relu(inverse)
            inverse = self.inverse_FC2(inverse)
            inverse = F.softmax(inverse)

            forward = self.forward_FC1(forward_input)
            forward = F.relu(forward)
            forward = self.forward_FC2(forward)

            return feature_st1, inverse, forward
