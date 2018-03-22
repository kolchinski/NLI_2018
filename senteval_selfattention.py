import numpy as np
import copy
from senteval import utils

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


'''
A = softmax(W2 tanh(W1 H^T))
'''


class SelfAttentionModel(nn.Module):

    def __init__(self, config):
        super(SelfAttentionModel, self).__init__()
        self.config = config

        self.num_units = self.config.hidden_size
        self.num_out_units = self.num_units * config.self_attn_outer_size
        if self.config.bidirectional:
            self.num_units *= 2

        # w1_selfattn = torch.FloatTensor(
        #     config.self_attn_inner_size, self.num_units
        # ).normal_(0, 0.1)
        # w1_selfattn = w1_selfattn.cuda()
        # w2_selfattn = torch.FloatTensor(
        #     config.self_attn_outer_size, config.self_attn_inner_size
        # ).normal_(0, 0.1)
        # w2_selfattn = w2_selfattn.cuda()
        # self.w1_selfattn = nn.Parameter(
        #     w1_selfattn, requires_grad=True)
        # self.w2_selfattn = nn.Parameter(
        #     w2_selfattn, requires_grad=True)

        self.w1_selfattn = nn.Linear(
            config.self_attn_inner_size, self.num_units)
        self.w2_selfattn = nn.Linear(
            config.self_attn_outer_size, config.self_attn_inner_size)

        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoder_outputs):
        encoder_out_tr = encoder_outputs.permute(0, 2, 1)  # [bs, nunits, slen]

        # pre_softmax = torch.matmul(
        #     self.w2_selfattn,
        #     self.tanh(torch.matmul(self.w1_selfattn, encoder_out_tr)),
        # )
        pre_softmax = \
            self.w2_selfattn(
                self.tanh(
                    self.w1_selfattn(encoder_out_tr)))
        A_selfattn = self.softmax(pre_softmax)

        # [bs, attn_outer_size, num_units]
        encoder_outputs_attn = torch.matmul(A_selfattn, encoder_outputs)

        # [bs, attn_outer_size * num_units]
        return encoder_outputs_attn.view(-1, self.num_out_units)
