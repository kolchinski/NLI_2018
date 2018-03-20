# adapted from https://github.com/pytorch/examples/blob/master/snli/model.py
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import src.models.transformer_pytorch as transformer_pytorch


class RNNEncoder(nn.Module):

    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config
        input_size = config.embedding_size
        if config.n_layers == 1:
            self.rnn = nn.LSTM(
                input_size=input_size, hidden_size=config.hidden_size,
                num_layers=config.n_layers, dropout=config.dp_ratio,
                bidirectional=config.bidirectional)
        elif config.n_layers == 2:
            self.rnn = nn.LSTM(
                input_size=input_size, hidden_size=config.layer1_hidden_size,
                num_layers=1, dropout=config.dp_ratio,
                bidirectional=config.bidirectional)
            self.rnn2 = nn.LSTM(
                input_size=config.layer1_hidden_size*2, hidden_size=config.hidden_size,
                num_layers=1, dropout=config.dp_ratio,
                bidirectional=config.bidirectional)

    def initHidden(self, batch_size):
        if self.config.bidirectional:
            state_shape = 2, batch_size, self.config.hidden_size
        else:
            state_shape = 1, batch_size, self.config.hidden_size
        h0 = c0 = Variable(torch.zeros(state_shape))
        return (h0, c0)

    def forward(self, inputs, hidden, batch_size):
        if self.config.n_layers == 2:
            outputs, (ht, ct) = self.rnn(inputs)
            outputs, (ht, ct) = self.rnn2(outputs)
        else:
            outputs, (ht, ct) = self.rnn(inputs, hidden)
        return outputs


class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config


class SiameseClassifierSentEmbed(nn.Module):

    def __init__(self, config, embed, encoder):
        super(SiameseClassifierSentEmbed, self).__init__()
        self.config = config
        self.embed = embed
        self.encoder = encoder

    def forward(
        self,
        encoder_input,
        encoder_len,
        encoder_pos_emb_input,
        encoder_unsort,
        encoder_init_hidden,
        batch_size
    ):
        prem_embed = encoder_input

        if self.config.encoder_type == 'transformer':
            premise = self.encoder(  # [max_len, batch_size, d_model]
                src_seq=encoder_input,
                src_pos=encoder_pos_emb_input,
            )
        elif self.config.encoder_type == 'rnn':
            premise = self.encoder(
                inputs=prem_embed,
                hidden=encoder_init_hidden,
                batch_size=batch_size
            )
            mask_value = -np.infty if self.config.sent_embed_type == 'maxpool' \
                else 0
            premise = nn.utils.rnn.pad_packed_sequence(
                premise, padding_value=mask_value)[0]
            premise = premise.index_select(1, encoder_unsort)

        len = premise.size(0)
        mask = Variable(torch.zeros(premise.size()))
        mask = mask.byte()
        if self.config.cuda:
            mask = mask.cuda()
        for i, _ in enumerate(encoder_len.data):
            l = encoder_len.data[i]
            if l < len:
                mask[l:, i, :] = 1
        premise[mask] = 0

        if self.config.sent_embed_type == 'maxpool':
            premise_sent_embed = torch.max(premise, dim=0)[0]  # [batch_size, embed_size]
        elif self.config.sent_embed_type == 'meanpool':
            premise_sent_embed = torch.div(
                torch.sum(premise, dim=0),
                Variable(encoder_len.data.unsqueeze(1)).float(),
            )
        elif self.config.sent_embed_type == 'mix':
            premise_max = torch.max(premise, dim=0)[0]
            premise_mean = torch.div(
                torch.sum(premise, dim=0),
                Variable(encoder_len.data.unsqueeze(1)).float(),
            )
            premise_sent_embed = torch.cat([premise_max,premise_mean],1)

        return premise_sent_embed


class SiameseClassifier(nn.Module):

    def __init__(self, config):
        super(SiameseClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.embedding_size)
        if self.config.fix_emb:
            self.embed.weight.requires_grad = False

        if config.encoder_type == 'transformer':
            self.encoder = transformer_pytorch.Encoder(
                n_src_vocab=config.n_embed,
                n_max_seq=config.max_length,
                src_word_emb=self.embed,
                wordemb_dim=self.config.embedding_size,
            )
        elif config.encoder_type == 'rnn':
            self.encoder = RNNEncoder(config)
        else:
            raise Exception("encoder_type not here {}".format(
                config.encoder_type))

        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()

        seq_in_size = 4 * config.hidden_size
        if self.config.bidirectional:
            seq_in_size *= 2

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
            nn.Linear(prev_hidden_size, config.d_out)
        )
        self.out = nn.Sequential(*classifier_transforms)

    def forward(
        self,
        encoder_input,
        encoder_pos_emb_input,
        encoder_unsort,
        decoder_input,
        decoder_pos_emb_input,
        decoder_unsort,
        encoder_init_hidden,
        batch_size
    ):
        prem_embed = encoder_input
        hypo_embed = decoder_input

        if self.config.encoder_type == 'transformer':
            premise = self.encoder(  # [max_len, batch_size, d_model]
                src_seq=encoder_input,
                src_pos=encoder_pos_emb_input,
            )
        else:
            premise = self.encoder(
                inputs=prem_embed,
                hidden=encoder_init_hidden,
                batch_size=batch_size
            )
            premise = nn.utils.rnn.pad_packed_sequence(premise)[0]
            premise = premise.index_select(1, encoder_unsort)
        if self.config.encoder_type == 'transformer':
            hypothesis = self.encoder(
                src_seq=decoder_input,
                src_pos=decoder_pos_emb_input,
            )
        else:
            hypothesis = self.encoder(
                inputs=hypo_embed,
                hidden=encoder_init_hidden,
                batch_size=batch_size
            )
            hypothesis = nn.utils.rnn.pad_packed_sequence(hypothesis)[0]
            hypothesis = hypothesis.index_select(1, decoder_unsort)

        premise_maxpool = torch.max(premise, 0)[0]  # [batch_size, embed_size]
        hypothesis_maxpool = torch.max(hypothesis, 0)[0]

        scores = self.out(torch.cat([
            premise_maxpool,
            hypothesis_maxpool,
            torch.abs(premise_maxpool - hypothesis_maxpool),
            premise_maxpool * hypothesis_maxpool,
        ], 1))  # [batch_size, 3]

        softmax_outputs = F.log_softmax(scores, dim=1)  # [batch_size, 3]

        return softmax_outputs
