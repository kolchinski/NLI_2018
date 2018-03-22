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

    def __init__(self, config, embed, encoder):
        super(SelfAttentionModel, self).__init__()
        self.config = config
        self.embed = embed
        self.encoder = encoder

        self.num_units = self.config.hidden_size
        self.num_out_units = self.num_units * config.self_attn_outer_size
        if self.config.bidirectional:
            self.num_units *= 2
        self.w1_selfattn = nn.Parameter(
            torch.FloatTensor(
                config.self_attn_inner_size, self.num_units
            ).normal_(0, 0.1),
            requires_grad=True,
        )
        self.w2_selfattn = nn.Parameter(
            torch.FloatTensor(
                config.self_attn_outer_size, config.self_attn_inner_size
            ).normal_(0, 0.1),
            requires_grad=True,
        )

    def forward(self, encoder_outputs, batch_len):
        encoder_out_batchfirst = encoder_outputs.permute(1, 0, 2)
        encoder_out_tr = encoder_outputs.permute(2, 0, 1)  # [bs, nunits, slen]

        pre_softmax = torch.matmul(
            self.s2w2_selfattn,
            torch.nn.Tanh(torch.matmul(self.w1_selfattn, encoder_out_tr)),
        )
        A_selfattn = torch.Softmax(pre_softmax, dim=1)

        # [bs, attn_outer_size, num_units]
        encoder_outputs_attn = torch.matmul(A_selfattn, encoder_out_batchfirst)

        # [bs, attn_outer_size * num_units]
        return encoder_outputs_attn.view(-1, self.num_out_units)
