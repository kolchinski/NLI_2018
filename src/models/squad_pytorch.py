import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import src.models.model_utils_pytorch as model_utils_pytorch


class SquadClassifier(nn.Module):

    def __init__(self, config, encoder, bottle=None):
        super(SquadClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.embedding_size)
        if self.config.fix_emb:
            self.embed.weight.requires_grad = False
        self.encoder = encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()

        seq_in_size = 7 * config.hidden_size
        if self.config.bidirectional:
            seq_in_size *= 2
        if bottle:
            self.bottle = bottle
            seq_in_size = config.bottle_dim
        elif hasattr(config, 'bottle_dim'):
            self.bottle = model_utils_pytorch.BottleLinear(
                d_in=seq_in_size, d_out=config.bottle_dim)
            seq_in_size = config.bottle_dim

        classifier_transforms = []
        prev_hidden_size = seq_in_size
        for next_hidden_size in config.mlp_classif_hidden_size_list:
            classifier_transforms.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                self.relu,
                self.dropout,
            ])
            prev_hidden_size = next_hidden_size
        classifier_transforms.append(
            nn.Linear(prev_hidden_size, config.d_out)  # d_out=2
        )
        self.out = nn.Sequential(*classifier_transforms)

    def forward(
        self,
        a1_packed_tensor,
        a1_idx_unsort,
        a2_packed_tensor,
        a2_idx_unsort,
        q_packed_tensor,
        q_idx_unsort,
        encoder_init_hidden,
        batch_size,
    ):
        if self.config.encoder_type == 'rnn':
            a1 = self.encoder(
                inputs=a1_packed_tensor,
                hidden=encoder_init_hidden,
                batch_size=batch_size,
            )
            a1 = nn.utils.rnn.pad_packed_sequence(
                a1, padding_value=-np.infty)[0]
            a1 = a1.index_select(1, a1_idx_unsort)

            a2 = self.encoder(
                inputs=a2_packed_tensor,
                hidden=encoder_init_hidden,
                batch_size=batch_size,
            )
            a2 = nn.utils.rnn.pad_packed_sequence(
                a2, padding_value=-np.infty)[0]
            a2 = a2.index_select(1, a2_idx_unsort)

            q = self.encoder(
                inputs=q_packed_tensor,
                hidden=encoder_init_hidden,
                batch_size=batch_size,
            )
            q = nn.utils.rnn.pad_packed_sequence(
                q, padding_value=-np.infty)[0]
            q = q.index_select(1, q_idx_unsort)
        else:
            raise Exception("{} not supported".format(
                self.config.encoder_type))

        a1_maxpool = torch.max(a1, 0)[0]  # [batch_size, embed_size]
        a2_maxpool = torch.max(a2, 0)[0]  # [batch_size, embed_size]
        q_maxpool = torch.max(q, 0)[0]

        if hasattr(self, 'bottle'):
            a1_maxpool = self.bottle(a1_maxpool)
            a2_maxpool = self.bottle(a2_maxpool)
            q_maxpool = self.bottle(q_maxpool)

        scores = self.out(torch.cat([
            q_maxpool,
            a1_maxpool,
            a2_maxpool,
            torch.abs(q_maxpool - a1_maxpool),
            q_maxpool * a1_maxpool,
            torch.abs(q_maxpool - a2_maxpool),
            q_maxpool * a2_maxpool,
        ], 1))  # [batch_size, 3]

        softmax_outputs = F.log_softmax(scores, dim=1)  # [batch_size, 2]

        return softmax_outputs
