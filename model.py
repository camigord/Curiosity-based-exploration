import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()

        # Policy network:
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        #############################################################
        # ICM Module
        self.icm_conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.icm_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.inverse_FC1 = nn.Linear(288*2, 256)
        self.inverse_FC2 = nn.Linear(256, num_outputs)

        self.forward_FC1 = nn.Linear(288 + num_outputs, 256)
        self.forward_FC2 = nn.Linear(256, 288)
        ##############################################################

        # Initialization
        self.apply(weights_init)
        self.inverse_FC1.weight.data = normalized_columns_initializer(
            self.inverse_FC1.weight.data, 0.01)
        self.inverse_FC1.bias.data.fill_(0)
        self.inverse_FC2.weight.data = normalized_columns_initializer(
            self.inverse_FC2.weight.data, 1.0)
        self.inverse_FC2.bias.data.fill_(0)

        self.forward_FC1.weight.data = normalized_columns_initializer(
            self.forward_FC1.weight.data, 0.01)
        self.forward_FC1.bias.data.fill_(0)
        self.forward_FC2.weight.data = normalized_columns_initializer(
            self.forward_FC2.weight.data, 1.0)
        self.forward_FC2.bias.data.fill_(0)
        ################################################################


        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()


    def forward(self, inputs, icm):

        if icm == False:
            """A3C"""
            inputs, (hx, cx) = inputs

            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            x = x.view(-1, 32 * 3 * 3)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx

            critic = self.critic_linear(x)
            actor = self.actor_linear(x)
            return critic, actor, (hx, cx)

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
